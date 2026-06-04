
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import numpy as np

# ─── Classification thresholds ───────────────────────────────────────────────
# Temperature (theo MAX30205 datasheet + lâm sàng)
TEMP_FEVER_THRESHOLD = 39.5   # °C - Sot (cong thuc)
TEMP_LOW_THRESHOLD = 32.5     # °C - Low temp (hypothermia)

# Heart Rate (theo công thức)
BPM_HIGH_THRESHOLD = 120.0    # beats per minute - Nguy hiểm
BPM_LOW_THRESHOLD = 50.0      # beats per minute - Thap
BPM_DELTA_THRESHOLD = 20.0    # Δ BPM > 20 → Bất thường

# SpO₂ (độ bão hòa oxy)
SPO2_LOW_THRESHOLD = 93.0   # % - Thap (canh bao)

# ─── Result labels ────────────────────────────────────────────────────────────
STATUS_FEVER = "FEVER"
STATUS_HIGH_HEART_RATE = "HIGH_HEART_RATE"
STATUS_LOW_HEART_RATE = "LOW_HEART_RATE"
STATUS_LOW_SPO2 = "LOW_SPO2"
STATUS_LOW_TEMP = "LOW_TEMP"
STATUS_FALL_DETECTED = "FALL_DETECTED"  # ← Trạng thái mới (ngã phát hiện)
STATUS_NORMAL = "NORMAL"

# ─── Batch parsing defaults ───────────────────────────────────────────────────
DEFAULT_BATCH_SAMPLE_INTERVAL_MS = int(os.getenv("BATCH_SAMPLE_INTERVAL_MS", "20"))
MIN_BPM = 35.0
MAX_BPM = 220.0
MIN_SPO2 = 70.0
MAX_SPO2 = 100.0

# PPG window / smoothing (aligned with firmware SPO2_WINDOW_SIZE=400, step=200)
MIN_PPG_SAMPLES_FOR_ESTIMATE = 80
VITAL_EMA_ALPHA_BPM = 0.18
VITAL_EMA_ALPHA_SPO2 = 0.12
VITAL_EMA_ALPHA_TEMP = 0.25
VITAL_MAX_REL_CHANGE_BPM = 0.12
VITAL_MAX_ABS_CHANGE_SPO2 = 2.5

# SpO2 calibration (MAX3010x empirical; align with firmware checkPPGQuality)
SPO2_CAL_A = float(os.getenv("SPO2_CAL_A", "110"))
SPO2_CAL_B = float(os.getenv("SPO2_CAL_B", "12"))

# Timestamp gap multiplier vs median sample interval → split PPG segment
PPG_GAP_INTERVAL_MULTIPLIER = 2.5
# Rolling-baseline deviation → invalidate motion/DC-step artifact (e.g. finger press)
PPG_DC_STEP_INVALIDATE_PCT = 0.035


# ─── Data structures ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HealthData:
    """Normalized telemetry data received from device/app."""

    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    heart_rate: float
    spo2: float | None
    temp: float
    ir: int | None
    red: int | None
    timestamp: int

    @property
    def status(self) -> str:
        return classify(self.heart_rate, self.temp, self.spo2)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        payload["bpm"] = int(round(payload["heart_rate"])) if payload["heart_rate"] is not None else None
        payload["ts"] = payload["timestamp"]
        return payload


# ─── Primitive conversion helpers ────────────────────────────────────────────
def _to_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Truong '{field_name}' khong hop le, can kieu so") from exc


def _to_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Truong '{field_name}' khong hop le, can so nguyen") from exc


def _to_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _to_float(value, field_name)


def _to_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _to_int(value, field_name)


def _to_float_list(value: Any, field_name: str) -> List[float]:
    if not isinstance(value, list):
        raise ValueError(f"Truong '{field_name}' khong hop le, can mang JSON")
    if not value:
        raise ValueError(f"Truong '{field_name}' khong duoc rong")

    return [_to_float(item, field_name) for item in value]


def _to_optional_int_list(value: Any, field_name: str, expected_len: int) -> List[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Truong '{field_name}' khong hop le, can mang JSON")
    
    # Nếu độ dài không khớp, cắt hoặc pad với 0
    if len(value) != expected_len:
        if len(value) > expected_len:
            value = value[:expected_len]
        else:
            value = value + [0] * (expected_len - len(value))

    return [_to_int(item, field_name) for item in value]


# ─── Signal helpers ──────────────────────────────────────────────────────────
def _first_batch_list_length(payload: Dict[str, Any], keys: List[str]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Truong '{key}' khong duoc rong")
            return len(value)
    return None


def _is_valid_vital(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def _sample_rate_hz(sample_interval_ms: int) -> float:
    return 1000.0 / float(max(1, sample_interval_ms))


def smooth_vital_ema(
    current: float | None,
    ema: float | None,
    alpha: float,
    *,
    max_rel_change: float | None = None,
    max_abs_change: float | None = None,
) -> tuple[float | None, float | None]:
    """
    EMA with optional outlier gate vs previous smoothed value.
    Returns (value_for_display, updated_ema_state).
    """
    if current is None:
        return ema, ema

    current_f = float(current)
    if ema is None:
        return current_f, current_f

    ema_f = float(ema)
    if max_rel_change is not None and ema_f > 0:
        rel = abs(current_f - ema_f) / ema_f
        if rel > max_rel_change:
            return ema_f, ema_f

    if max_abs_change is not None:
        if abs(current_f - ema_f) > max_abs_change:
            return ema_f, ema_f

    updated = alpha * current_f + (1.0 - alpha) * ema_f
    return updated, updated


def _infer_sample_interval_ms(
    normalized: Dict[str, Any],
    sample_count: int,
    fallback_ms: int,
) -> int:
    explicit_interval = normalized.get("sample_interval_ms")
    if explicit_interval is not None:
        return max(1, _to_int(explicit_interval, "sample_interval_ms"))

    if sample_count > 1 and normalized.get("ts0") is not None:
        try:
            end_ts = _to_int(
                normalized.get("timestamp", normalized.get("ts", int(time.time() * 1000))),
                "timestamp",
            )
            start_ts = _to_int(normalized["ts0"], "ts0")
            duration_ms = max(1, end_ts - start_ts)
            inferred = int(round(duration_ms / float(sample_count - 1)))
            if inferred > 0:
                return inferred
        except ValueError:
            pass

    return max(1, int(fallback_ms))


def _sanitize_optical_series(values: List[int]) -> List[int]:
    if not values:
        return values
    # Drop invalid placeholders early to stabilize peak/AC calculations.
    return [int(v) if v is not None and int(v) > 0 else 0 for v in values]


def _split_contiguous_valid_segments(values: List[int] | None, min_valid_value: int = 1) -> list:
    """Return list of (start_index, segment_values) for contiguous runs where value >= min_valid_value.

    If `values` is None, returns empty list.
    """
    if values is None:
        return []
    segments = []
    start = None
    buf = []
    for i, v in enumerate(values):
        try:
            valid = (v is not None) and (int(v) >= min_valid_value)
        except Exception:
            valid = False

        if valid:
            if start is None:
                start = i
                buf = [int(v)]
            else:
                buf.append(int(v))
        else:
            if start is not None:
                segments.append((start, buf))
                start = None
                buf = []

    if start is not None and buf:
        segments.append((start, buf))

    return segments


def _invalidate_ppg_gaps_and_artifacts(
    values: List[int | None],
    sample_interval_ms: int,
    timestamps: List[int] | None = None,
) -> List[int | None]:
    """Mark samples invalid at timing gaps and large DC steps (motion artifacts)."""
    if not values:
        return values

    n = len(values)
    invalid = [False] * n

    if timestamps and len(timestamps) == n:
        dts: List[int] = []
        for i in range(1, n):
            if timestamps[i] is None or timestamps[i - 1] is None:
                continue
            dt = int(timestamps[i]) - int(timestamps[i - 1])
            if dt > 0:
                dts.append(dt)
        gap_ms = max(50, int(sample_interval_ms * PPG_GAP_INTERVAL_MULTIPLIER))
        if dts:
            median_dt = int(np.median(np.array(dts, dtype=np.int64)))
            gap_ms = max(gap_ms, int(median_dt * PPG_GAP_INTERVAL_MULTIPLIER))
        for i in range(1, n):
            if timestamps[i] is None or timestamps[i - 1] is None:
                continue
            if int(timestamps[i]) - int(timestamps[i - 1]) > gap_ms:
                invalid[i] = True

    arr = np.array(
        [float(v) if v is not None and int(v) > 0 else np.nan for v in values],
        dtype=np.float64,
    )
    window = max(5, int(1000.0 / max(1, sample_interval_ms) * 0.35))
    for i in range(n):
        if invalid[i] or not np.isfinite(arr[i]):
            continue
        start = max(0, i - window)
        baseline = np.nanmedian(arr[start:i]) if i > start else arr[i]
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        if abs(arr[i] - baseline) / baseline > PPG_DC_STEP_INVALIDATE_PCT:
            invalid[i] = True

    # Sharp DC step: drop >6% vs local median in ~0.5s (finger press), not normal PPG ripple
    step_window = max(5, int(1000.0 / max(1, sample_interval_ms) * 0.5))
    for i in range(n):
        if invalid[i] or not np.isfinite(arr[i]):
            continue
        start = max(0, i - step_window)
        local = arr[start : i + 1]
        local = local[np.isfinite(local)]
        if local.size < 3:
            continue
        baseline = float(np.median(local))
        if baseline > 0 and (baseline - float(arr[i])) / baseline > 0.06:
            invalid[i] = True

    out: List[int | None] = []
    for i, v in enumerate(values):
        if invalid[i] or v is None or int(v) <= 0:
            out.append(None)
        else:
            out.append(int(v))
    return out


def _remove_dc(signal: np.ndarray, window_size: int) -> np.ndarray:
    if window_size < 2 or signal.size == 0:
        return signal - float(np.mean(signal))
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    baseline = np.convolve(signal, kernel, mode="same")
    return signal - baseline


def _estimate_bpm_autocorr(signal: np.ndarray, sample_rate_hz: float) -> float | None:
    if signal.size < 40:
        return None

    centered = signal.astype(np.float64) - float(np.mean(signal))
    std = float(np.std(centered))
    if std <= 1e-6:
        return None
    centered = centered / std

    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    min_lag = max(1, int(sample_rate_hz * 60.0 / MAX_BPM))
    max_lag = min(len(corr) - 1, int(sample_rate_hz * 60.0 / MIN_BPM))
    if max_lag <= min_lag:
        return None

    segment = corr[min_lag : max_lag + 1]
    if segment.size == 0:
        return None

    peak_lag: int | None = None
    max_v = float(np.max(segment))
    for i in range(1, len(segment) - 1):
        if segment[i] < segment[i - 1] or segment[i] < segment[i + 1]:
            continue
        if segment[i] < 0.5 * max_v:
            continue
        peak_lag = min_lag + i
        break

    if peak_lag is None:
        peak_lag = min_lag + int(np.argmax(segment))

    if peak_lag <= 0:
        return None

    bpm = 60.0 * sample_rate_hz / float(peak_lag)
    return float(np.clip(bpm, MIN_BPM, MAX_BPM))


def _filter_rr_intervals_iqr(rr_seconds: np.ndarray) -> np.ndarray:
    if rr_seconds.size < 2:
        return rr_seconds

    q1, q3 = np.percentile(rr_seconds, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return rr_seconds

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return rr_seconds[(rr_seconds >= low) & (rr_seconds <= high)]


def _bandpass_fft(signal: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(signal)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=signal.size)


def _estimate_bpm_fft(signal: np.ndarray, sample_rate_hz: float) -> float | None:
    """Dominant frequency in cardiac band (robust on short MAX3010x windows)."""
    if signal.size < 40:
        return None

    centered = signal.astype(np.float64) - float(np.mean(signal))
    if float(np.std(centered)) <= 1e-6:
        return None

    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    mask = (freqs >= 0.75) & (freqs <= 2.5)
    if not np.any(mask):
        return None

    band_freqs = freqs[mask]
    band_mag = spectrum[mask]
    peak_mag = float(np.max(band_mag))
    if peak_mag <= 0:
        return None

    # Prefer fundamental (lowest strong spectral line), not 2× harmonic.
    strong_freqs: List[float] = []
    for i in range(1, len(band_mag) - 1):
        if band_mag[i] < band_mag[i - 1] or band_mag[i] < band_mag[i + 1]:
            continue
        if band_mag[i] < 0.55 * peak_mag:
            continue
        strong_freqs.append(float(band_freqs[i]))

    if strong_freqs:
        peak_f = min(strong_freqs)
    else:
        peak_f = float(band_freqs[int(np.argmax(band_mag))])

    if peak_f <= 0:
        return None

    return float(np.clip(peak_f * 60.0, MIN_BPM, MAX_BPM))


def _resolve_bpm_estimates(estimates: List[float]) -> float | None:
    if not estimates:
        return None

    in_range = [e for e in estimates if BPM_LOW_THRESHOLD <= e <= BPM_HIGH_THRESHOLD]
    pool = in_range if in_range else estimates

    # Sub-harmonic fix: if all estimates are too low, try doubling once
    if pool and max(pool) < 55.0:
        doubled = [e * 2.0 for e in pool if e * 2.0 <= MAX_BPM]
        doubled_in = [e for e in doubled if BPM_LOW_THRESHOLD <= e <= BPM_HIGH_THRESHOLD]
        if doubled_in:
            pool = doubled_in

    # Supra-harmonic fix: if estimate > 100, try half
    if pool and min(pool) > 100.0:
        halved = [e / 2.0 for e in pool if e / 2.0 >= MIN_BPM]
        halved_in = [e for e in halved if BPM_LOW_THRESHOLD <= e <= BPM_HIGH_THRESHOLD]
        if halved_in:
            pool = halved_in

    return float(np.median(np.array(pool, dtype=np.float64)))


def _estimate_bpm_from_ir_segment(signal: np.ndarray, sample_rate_hz: float) -> float | None:
    if signal.size < 20:
        return None

    signal_std = float(np.std(signal))
    mean_signal = float(np.mean(np.abs(signal)))
    rel_std = signal_std / mean_signal if mean_signal > 0 else 0.0
    if mean_signal <= 1e-6 or rel_std < 1e-4:
        return None

    dc_window = max(2, int(sample_rate_hz * 0.5))
    centered = _remove_dc(signal, dc_window)
    filtered = _bandpass_fft(centered, sample_rate_hz, 0.5, 4.0)

    filtered_std = float(np.std(filtered))
    if filtered_std <= 1e-6:
        return None

    estimates: List[float] = []
    bpm_fft = _estimate_bpm_fft(filtered, sample_rate_hz)
    if bpm_fft is not None:
        estimates.append(bpm_fft)

    normalized = filtered / filtered_std
    threshold = max(0.25, float(np.percentile(normalized, 65)))
    min_gap = max(1, int(sample_rate_hz * 0.35))

    peaks: List[int] = []
    for idx in range(1, len(normalized) - 1):
        is_peak = normalized[idx] > normalized[idx - 1] and normalized[idx] >= normalized[idx + 1]
        if not is_peak or normalized[idx] <= threshold:
            continue

        if not peaks or idx - peaks[-1] >= min_gap:
            peaks.append(idx)
            continue

        if centered[idx] > centered[peaks[-1]]:
            peaks[-1] = idx

    bpm_ac = _estimate_bpm_autocorr(filtered, sample_rate_hz)
    if bpm_ac is not None:
        estimates.append(bpm_ac)

    if len(peaks) >= 2:
        rr_seconds = np.diff(peaks) / sample_rate_hz
        min_rr = 60.0 / MAX_BPM
        max_rr = 60.0 / MIN_BPM
        rr_seconds = rr_seconds[(rr_seconds > min_rr) & (rr_seconds < max_rr)]
        if rr_seconds.size >= 2:
            rr_seconds = _filter_rr_intervals_iqr(rr_seconds)
        if rr_seconds.size > 0:
            estimates.append(60.0 / float(np.median(rr_seconds)))

    return _resolve_bpm_estimates(estimates)


def _segment_cardiac_score(signal: np.ndarray, sample_rate_hz: float) -> float:
    if signal.size < 20:
        return 0.0
    dc_window = max(2, int(sample_rate_hz * 0.5))
    filtered = _bandpass_fft(_remove_dc(signal, dc_window), sample_rate_hz, 0.5, 4.0)
    mean_val = float(np.mean(np.abs(signal)))
    if mean_val <= 0:
        return 0.0
    return float(np.std(filtered)) / mean_val


def _estimate_bpm_from_ir(ir_values: List[int], sample_interval_ms: int) -> float | None:
    if not ir_values:
        return None

    segments = _split_contiguous_valid_segments(ir_values, min_valid_value=1)
    if not segments:
        return None

    sample_rate_hz = _sample_rate_hz(sample_interval_ms)
    scored: List[tuple[float, float]] = []

    win = max(40, int(sample_rate_hz * 1.7))
    stride = max(8, win // 4)

    for _start, seg in segments:
        if len(seg) < 40:
            continue
        arr = np.array(seg, dtype=np.float64)
        window_bpms: List[float] = []

        if len(seg) >= win:
            for offset in range(0, len(seg) - win + 1, stride):
                chunk = arr[offset : offset + win]
                peak = float(np.max(chunk))
                trough = float(np.min(chunk))
                if peak > 0 and (peak - trough) / peak > 0.035:
                    continue
                bpm = _estimate_bpm_from_ir_segment(chunk, sample_rate_hz)
                if bpm is not None:
                    window_bpms.append(bpm)
        else:
            peak = float(np.max(arr))
            trough = float(np.min(arr))
            if peak > 0 and (peak - trough) / peak <= 0.035:
                bpm = _estimate_bpm_from_ir_segment(arr, sample_rate_hz)
                if bpm is not None:
                    window_bpms.append(bpm)

        bpm = _resolve_bpm_estimates(window_bpms)
        if bpm is None:
            continue
        score = _segment_cardiac_score(arr, sample_rate_hz)
        scored.append((score, bpm))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:2]
    bpms = [b for _s, b in top if BPM_LOW_THRESHOLD <= b <= BPM_HIGH_THRESHOLD]
    if not bpms:
        bpms = [b for _s, b in top]
    return _resolve_bpm_estimates(bpms)


def _estimate_spo2_from_acdc_ratio(ratio: float) -> float | None:
    if ratio <= 0 or not np.isfinite(ratio):
        return None
    ratio = float(np.clip(ratio, 0.4, 1.6))
    spo2 = SPO2_CAL_A - SPO2_CAL_B * ratio
    return float(np.clip(spo2, MIN_SPO2, MAX_SPO2))


def _estimate_spo2_from_quality(quality: Dict[str, Any]) -> float | None:
    """Use ir_acdc/red_acdc from firmware (same window, peak-to-peak / mean)."""
    if not isinstance(quality, dict) or not quality.get("valid", False):
        return None
    try:
        ir_acdc = float(quality["ir_acdc"])
        red_acdc = float(quality["red_acdc"])
    except (KeyError, TypeError, ValueError):
        return None
    if ir_acdc <= 0:
        return None
    return _estimate_spo2_from_acdc_ratio(red_acdc / ir_acdc)


def _estimate_spo2_from_p2p(ir_values: List[int], red_values: List[int]) -> float | None:
    """Peak-to-peak / mean — matches ESP32 checkPPGQuality()."""
    ir_valid = [int(v) for v in ir_values if v is not None and int(v) > 0]
    red_valid = [int(v) for v in red_values if v is not None and int(v) > 0]
    if len(ir_valid) < 20 or len(red_valid) < 20:
        return None

    ir_arr = np.array(ir_valid, dtype=np.float64)
    red_arr = np.array(red_valid, dtype=np.float64)
    ir_dc = float(np.mean(ir_arr))
    red_dc = float(np.mean(red_arr))
    if ir_dc <= 0 or red_dc <= 0:
        return None

    ir_acdc = (float(np.max(ir_arr)) - float(np.min(ir_arr))) / ir_dc
    red_acdc = (float(np.max(red_arr)) - float(np.min(red_arr))) / red_dc
    if ir_acdc <= 0:
        return None
    return _estimate_spo2_from_acdc_ratio(red_acdc / ir_acdc)


def _estimate_spo2_from_ir_red(
    ir_values: List[int],
    red_values: List[int],
    sample_interval_ms: int,
) -> float | None:
    """
    Estimate SpO2 using AC/DC ratio (p2p/mean preferred; bandpass fallback).
    """
    # Find overlapping valid contiguous segments where both IR and RED >=1
    ir_segs = _split_contiguous_valid_segments(ir_values, min_valid_value=1)
    red_segs = _split_contiguous_valid_segments(red_values, min_valid_value=1)
    if not ir_segs or not red_segs:
        return None

    # Build list of overlapping segments (start index relative to original arrays)
    best_segment = None
    best_len = 0
    for i_start, i_seg in ir_segs:
        i_end = i_start + len(i_seg) - 1
        for r_start, r_seg in red_segs:
            r_end = r_start + len(r_seg) - 1
            # overlap region
            overlap_start = max(i_start, r_start)
            overlap_end = min(i_end, r_end)
            overlap_len = overlap_end - overlap_start + 1
            if overlap_len >= 20:
                if overlap_len > best_len:
                    # extract overlapping slices
                    ir_slice = [int(v) for v in ir_values[overlap_start:overlap_end+1]]
                    red_slice = [int(v) for v in red_values[overlap_start:overlap_end+1]]
                    best_segment = (overlap_start, ir_slice, red_slice)
                    best_len = overlap_len
    
    if best_segment is None:
        return _estimate_spo2_from_p2p(ir_values, red_values)

    _overlap_start, ir_slice, red_slice = best_segment
    p2p_spo2 = _estimate_spo2_from_p2p(ir_slice, red_slice)
    if p2p_spo2 is not None:
        return p2p_spo2

    ir = np.array(ir_slice, dtype=np.float64)
    red = np.array(red_slice, dtype=np.float64)
    sample_rate_hz = _sample_rate_hz(sample_interval_ms)
    dc_window = max(2, int(sample_rate_hz * 0.5))
    ir_dc = float(np.mean(ir))
    red_dc = float(np.mean(red))
    ir_filt = _bandpass_fft(_remove_dc(ir, dc_window), sample_rate_hz, 0.5, 4.0)
    red_filt = _bandpass_fft(_remove_dc(red, dc_window), sample_rate_hz, 0.5, 4.0)
    ir_ac = float(np.std(ir_filt))
    red_ac = float(np.std(red_filt))
    if ir_dc <= 1e-6 or red_dc <= 1e-6 or ir_ac <= 1e-6:
        return None
    return _estimate_spo2_from_acdc_ratio((red_ac / red_dc) / (ir_ac / ir_dc))


# ─── Payload normalization ───────────────────────────────────────────────────
def _is_batch_payload(payload: Dict[str, Any]) -> bool:
    series_keys = ("ax", "ay", "az", "gx", "gy", "gz", "ir", "red")
    return any(isinstance(payload.get(key), list) for key in series_keys)


def _normalize_payload_aliases(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)

    if "heart_rate" not in normalized and "bpm" in normalized:
        normalized["heart_rate"] = normalized["bpm"]
    if "timestamp" not in normalized and "ts" in normalized:
        normalized["timestamp"] = normalized["ts"]

    return normalized


def _build_health_data_sample(
    ax: float,
    ay: float,
    az: float,
    gx: float,
    gy: float,
    gz: float,
    heart_rate: float,
    spo2: float | None,
    temp: float,
    ir: int | None,
    red: int | None,
    timestamp: int,
) -> HealthData:
    return HealthData(
        ax=ax,
        ay=ay,
        az=az,
        gx=gx,
        gy=gy,
        gz=gz,
        heart_rate=heart_rate,
        spo2=spo2,
        temp=temp,
        ir=ir,
        red=red,
        timestamp=timestamp,
    )


# ─── Temperature filtering ──────────────────────────────────────────────────
# Median filter: T = median(T_{i-2}, T_{i-1}, T_i)
# EMA (Exponential Moving Average): T_smooth = 0.5T + 0.5T_old

_last_temperature = 36.5
_temperature_lock = None

def _median_filter_temperature(temps: List[float]) -> float:
    
    if not temps:
        return 36.5
    return float(np.median(np.array(temps, dtype=np.float64)))


def _ema_smooth_temperature(temp_new: float, temp_old: float = None, alpha: float = 0.5) -> float:
    """
    EMA (Exponential Moving Average): T_smooth = α×T_new + (1-α)×T_old
    
    Args:
        temp_new: Mẫu nhiệt độ mới từ MAX30205
        temp_old: Giá trị EMA trước đó (default: từ _last_temperature)
        alpha: Smooth factor (default: 0.5)
    
    Returns:
        Giá trị nhiệt độ đã làm mượt
    """
    global _last_temperature
    
    if temp_old is None:
        temp_old = _last_temperature
    
    temp_smooth = alpha * temp_new + (1.0 - alpha) * temp_old
    _last_temperature = temp_smooth

    return temp_smooth


# ─── Public payload parsers ───────────────────────────────────────────────────
def from_batch_dict(payload: Dict[str, Any]) -> List[HealthData]:
    """Parse batch telemetry payload into a list of normalized samples."""
    normalized = _normalize_payload_aliases(payload)

    batch_keys = ("ax", "ay", "az", "gx", "gy", "gz", "ir", "red")
    sample_count = _first_batch_list_length(normalized, list(batch_keys))
    if sample_count is None:
        raise ValueError("Payload batch khong co truong mang nao hop le")

    series_map: Dict[str, List[float]] = {}
    for key in ("ax", "ay", "az", "gx", "gy", "gz"):
        if key in normalized and normalized[key] is not None:
            values = _to_float_list(normalized[key], key)
            # Nếu độ dài không khớp, cắt hoặc pad
            if len(values) != sample_count:
                if len(values) > sample_count:
                    values = values[:sample_count]
                else:
                    values = values + [0.0] * (sample_count - len(values))
            series_map[key] = values
        else:
            series_map[key] = [0.0] * sample_count

    ir_series = None
    if normalized.get("ir") is not None:
        ir_series = _to_optional_int_list(normalized.get("ir"), "ir", sample_count)

    red_series = None
    if normalized.get("red") is not None:
        red_series = _to_optional_int_list(normalized.get("red"), "red", sample_count)

    sample_interval_ms = _infer_sample_interval_ms(
        normalized,
        sample_count=sample_count,
        fallback_ms=DEFAULT_BATCH_SAMPLE_INTERVAL_MS,
    )

    batch_end_timestamp = _to_int(
        normalized.get("timestamp", normalized.get("ts", int(time.time() * 1000))),
        "timestamp",
    )
    if normalized.get("ts0") is not None:
        batch_start_timestamp = _to_int(normalized["ts0"], "ts0")
    else:
        batch_start_timestamp = batch_end_timestamp - (sample_count - 1) * sample_interval_ms

    raw_temp = normalized.get("temp")
    if raw_temp is not None:
        temp_value = _ema_smooth_temperature(
            _to_float(raw_temp, "temp"),
            alpha=VITAL_EMA_ALPHA_TEMP,
        )
    else:
        temp_value = _last_temperature

    # Mark invalid optical samples (<=0) as None so estimators can skip them
    raw_ir_series = _sanitize_optical_series(ir_series) if ir_series is not None else None
    raw_red_series = _sanitize_optical_series(red_series) if red_series is not None else None

    marked_ir_series = None
    if raw_ir_series is not None:
        marked_ir_series = [v if (v is not None and int(v) > 0) else None for v in raw_ir_series]

    marked_red_series = None
    if raw_red_series is not None:
        marked_red_series = [v if (v is not None and int(v) > 0) else None for v in raw_red_series]

    ppg_timestamps = normalized.get("ppg_timestamps")
    if isinstance(ppg_timestamps, list) and marked_ir_series is not None:
        marked_ir_series = _invalidate_ppg_gaps_and_artifacts(
            marked_ir_series,
            sample_interval_ms,
            timestamps=ppg_timestamps,
        )
        if marked_red_series is not None:
            marked_red_series = _invalidate_ppg_gaps_and_artifacts(
                marked_red_series,
                sample_interval_ms,
                timestamps=ppg_timestamps,
            )

    heart_rate: float | None = None
    if _is_valid_vital(normalized.get("heart_rate")):
        heart_rate = _to_float(normalized["heart_rate"], "heart_rate")
    elif marked_ir_series is not None:
        heart_rate = _estimate_bpm_from_ir(marked_ir_series, sample_interval_ms)

    spo2_value: float | None = None
    if _is_valid_vital(normalized.get("spo2")):
        spo2_value = _to_float(normalized["spo2"], "spo2")
    else:
        quality = normalized.get("quality")
        if isinstance(quality, dict):
            spo2_value = _estimate_spo2_from_quality(quality)
        if spo2_value is None and marked_ir_series is not None and marked_red_series is not None:
            spo2_value = _estimate_spo2_from_ir_red(
                marked_ir_series, marked_red_series, sample_interval_ms
            )

    samples: List[HealthData] = []
    for idx in range(sample_count):
        sample_timestamp = batch_start_timestamp + idx * sample_interval_ms
        samples.append(
            _build_health_data_sample(
                ax=series_map["ax"][idx],
                ay=series_map["ay"][idx],
                az=series_map["az"][idx],
                gx=series_map["gx"][idx],
                gy=series_map["gy"][idx],
                gz=series_map["gz"][idx],
                heart_rate=heart_rate,
                spo2=spo2_value,
                temp=temp_value,
                ir=marked_ir_series[idx] if marked_ir_series is not None else None,
                red=marked_red_series[idx] if marked_red_series is not None else None,
                timestamp=sample_timestamp,
            )
        )

    return samples


def from_dict(payload: Dict[str, Any]) -> HealthData:
    """Parse and validate telemetry payload from a Python dict."""
    normalized = _normalize_payload_aliases(payload)

    # Older and simplified payloads may not include motion/optical fields.
    normalized.setdefault("ax", 0.0)
    normalized.setdefault("ay", 0.0)
    normalized.setdefault("az", 0.0)
    normalized.setdefault("gx", 0.0)
    normalized.setdefault("gy", 0.0)
    normalized.setdefault("gz", 0.0)
    normalized.setdefault("timestamp", int(time.time()))
    normalized.setdefault("heart_rate", 0.0)
    normalized.setdefault("spo2", None)
    normalized.setdefault("ir", None)
    normalized.setdefault("red", None)

    required = ("temp",)
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ValueError(f"Thieu truong bat buoc: {', '.join(missing)}")

    return _build_health_data_sample(
        ax=_to_float(normalized["ax"], "ax"),
        ay=_to_float(normalized["ay"], "ay"),
        az=_to_float(normalized["az"], "az"),
        gx=_to_float(normalized["gx"], "gx"),
        gy=_to_float(normalized["gy"], "gy"),
        gz=_to_float(normalized["gz"], "gz"),
        heart_rate=_to_float(normalized["heart_rate"], "heart_rate"),
        spo2=_to_optional_float(normalized.get("spo2"), "spo2"),
        temp=_to_float(normalized["temp"], "temp"),
        ir=_to_optional_int(normalized.get("ir"), "ir"),
        red=_to_optional_int(normalized.get("red"), "red"),
        timestamp=_to_int(normalized["timestamp"], "timestamp"),
    )


def from_json(raw: str) -> HealthData:
    """Backward-compatible parser that returns one sample."""
    samples = from_json_samples(raw)
    return samples[-1]


def from_json_samples(raw: str) -> List[HealthData]:
    """Parse JSON telemetry payload into one or more normalized samples."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Payload JSON khong hop le") from exc

    if not isinstance(payload, dict):
        raise ValueError("Payload khong hop le, can doi tuong JSON")

    if _is_batch_payload(payload):
        return from_batch_dict(payload)

    return [from_dict(payload)]


# ─── Classification ──────────────────────────────────────────────────────────
def classify(bpm, temp, spo2) -> List[str]:
    statuses = []

    if temp < TEMP_LOW_THRESHOLD:
        statuses.append(STATUS_LOW_TEMP)

    if temp > TEMP_FEVER_THRESHOLD:
        statuses.append(STATUS_FEVER)

    if bpm is not None:
        if bpm < BPM_LOW_THRESHOLD:
            statuses.append(STATUS_LOW_HEART_RATE)

        elif bpm > BPM_HIGH_THRESHOLD:
            statuses.append(STATUS_HIGH_HEART_RATE)

    if spo2 is not None:
        if spo2 < SPO2_LOW_THRESHOLD:
            statuses.append(STATUS_LOW_SPO2)

    if not statuses:
        statuses.append(STATUS_NORMAL)

    return statuses


# ─── Health Analytics Utils (feature/add-health-utilities) ──────────────────
# 
# def validate_health_data(data: HealthData) -> bool:
#     """
#     Validate health data completeness and consistency.
#     
#     Checks:
#     - Heart rate in valid range [MIN_BPM, MAX_BPM]
#     - Temperature in realistic range (25°C - 42°C)
#     - SpO2 in valid range [MIN_SPO2, MAX_SPO2] if present
#     - Accelerometer values not NaN
#     - Timestamp is positive
#     
#     Args:
#         data: HealthData object to validate
#         
#     Returns:
#         True if all validations pass, False otherwise
#     """
#     if data.heart_rate < MIN_BPM or data.heart_rate > MAX_BPM:
#         return False
#     if data.temp < 25.0 or data.temp > 42.0:
#         return False
#     if data.spo2 is not None:
#         if data.spo2 < MIN_SPO2 or data.spo2 > MAX_SPO2:
#             return False
#     if data.timestamp <= 0:
#         return False
#     if np.isnan([data.ax, data.ay, data.az]).any():
#         return False
#     return True


# def detect_anomalies(samples: List[HealthData]) -> Dict[str, List[int]]:
#     """
#     Detect anomalies in health data stream using statistical methods.
#     
#     Anomalies detected:
#     - Sudden BPM spike (>30 bpm/s change)
#     - Temperature fluctuation (>2°C in 60s)
#     - SpO2 drop (>5% in 30s)
#     - Accelerometer noise (RMS > threshold)
#     
#     Args:
#         samples: List of HealthData samples ordered by timestamp
#         
#     Returns:
#         Dictionary mapping anomaly_type -> list of sample indices
#     """
#     anomalies = {
#         "bpm_spike": [],
#         "temp_fluctuation": [],
#         "spo2_drop": [],
#         "accel_noise": []
#     }
#     
#     if len(samples) < 2:
#         return anomalies
#     
#     for i in range(1, len(samples)):
#         prev = samples[i-1]
#         curr = samples[i]
#         
#         # BPM spike detection
#         if curr.heart_rate and prev.heart_rate:
#             bpm_delta = abs(curr.heart_rate - prev.heart_rate)
#             if bpm_delta > 30:
#                 anomalies["bpm_spike"].append(i)
#         
#         # Temperature fluctuation
#         temp_delta = abs(curr.temp - prev.temp)
#         if temp_delta > 2.0:
#             anomalies["temp_fluctuation"].append(i)
#         
#         # SpO2 drop
#         if curr.spo2 and prev.spo2:
#             spo2_delta = prev.spo2 - curr.spo2
#             if spo2_delta > 5.0:
#                 anomalies["spo2_drop"].append(i)
#     
#     return anomalies


# def generate_health_report(samples: List[HealthData], duration_sec: int) -> Dict[str, Any]:
#     """
#     Generate comprehensive health analytics report from sample batch.
#     
#     Computes:
#     - Heart rate statistics (min, max, avg, std)
#     - Temperature statistics and trend
#     - SpO2 statistics and stability
#     - Fall detection events
#     - Anomaly summary
#     - Risk level assessment (LOW, MEDIUM, HIGH, CRITICAL)
#     
#     Args:
#         samples: List of HealthData samples
#         duration_sec: Time window duration in seconds
#         
#     Returns:
#         Dictionary with comprehensive health metrics
#     """
#     report = {
#         "duration_sec": duration_sec,
#         "sample_count": len(samples),
#         "timestamp_start": samples[0].timestamp if samples else None,
#         "timestamp_end": samples[-1].timestamp if samples else None,
#         "heart_rate": {},
#         "temperature": {},
#         "spo2": {},
#         "fall_events": 0,
#         "anomalies": {},
#         "risk_level": "UNKNOWN"
#     }
#     
#     if not samples:
#         return report
#     
#     bpm_values = [s.heart_rate for s in samples if s.heart_rate]
#     if bpm_values:
#         report["heart_rate"] = {
#             "min": float(np.min(bpm_values)),
#             "max": float(np.max(bpm_values)),
#             "avg": float(np.mean(bpm_values)),
#             "std": float(np.std(bpm_values))
#         }
#     
#     temp_values = [s.temp for s in samples]
#     report["temperature"] = {
#         "min": float(np.min(temp_values)),
#         "max": float(np.max(temp_values)),
#         "avg": float(np.mean(temp_values)),
#         "std": float(np.std(temp_values))
#     }
#     
#     spo2_values = [s.spo2 for s in samples if s.spo2]
#     if spo2_values:
#         report["spo2"] = {
#             "min": float(np.min(spo2_values)),
#             "max": float(np.max(spo2_values)),
#             "avg": float(np.mean(spo2_values)),
#             "std": float(np.std(spo2_values))
#         }
#     
#     # Count fall events
#     fall_count = sum(1 for s in samples if STATUS_FALL_DETECTED in s.status)
#     report["fall_events"] = fall_count
#     
#     # Risk assessment
#     if fall_count > 0:
#         report["risk_level"] = "CRITICAL"
#     elif report["temperature"]["max"] > TEMP_FEVER_THRESHOLD or report["temperature"]["min"] < TEMP_LOW_THRESHOLD:
#         report["risk_level"] = "HIGH"
#     elif spo2_values and min(spo2_values) < SPO2_LOW_THRESHOLD:
#         report["risk_level"] = "HIGH"
#     elif bpm_values and (min(bpm_values) < BPM_LOW_THRESHOLD or max(bpm_values) > BPM_HIGH_THRESHOLD):
#         report["risk_level"] = "MEDIUM"
#     else:
#         report["risk_level"] = "LOW"
#     
#     return report
