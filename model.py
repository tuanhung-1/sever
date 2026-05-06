

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

# ─── Classification thresholds ───────────────────────────────────────────────
# Temperature (theo MAX30205 datasheet + lâm sàng)
TEMP_FEVER_THRESHOLD = 37.5   # °C - Sốt (công thức)
TEMP_DANGER_THRESHOLD = 35.0  # °C - Hypothermia (nguy hiểm)

# Heart Rate (theo công thức)
BPM_HIGH_THRESHOLD = 120.0    # beats per minute - Nguy hiểm
BPM_LOW_THRESHOLD = 50.0      # beats per minute - Thấp
BPM_DELTA_THRESHOLD = 20.0    # Δ BPM > 20 → Bất thường

# SpO₂ (độ bão hòa oxy)
SPO2_WARNING_THRESHOLD = 95.0  # % - Cảnh báo
SPO2_DANGER_THRESHOLD = 90.0   # % - Nguy hiểm

# ─── Result labels ────────────────────────────────────────────────────────────
STATUS_FEVER = "FEVER"
STATUS_HIGH_HEART_RATE = "HIGH_HEART_RATE"
STATUS_LOW_HEART_RATE = "LOW_HEART_RATE"
STATUS_FALL_DETECTED = "FALL_DETECTED"  # ← Trạng thái mới (ngã phát hiện)
STATUS_NORMAL = "NORMAL"

# ─── Batch parsing defaults ───────────────────────────────────────────────────
DEFAULT_BATCH_SAMPLE_INTERVAL_MS = int(os.getenv("BATCH_SAMPLE_INTERVAL_MS", "20"))
MIN_BPM = 35.0
MAX_BPM = 220.0
MIN_SPO2 = 70.0
MAX_SPO2 = 100.0


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
        return classify(self.heart_rate, self.temp)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        payload["bpm"] = int(round(payload["heart_rate"]))
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
    if len(value) != expected_len:
        raise ValueError(f"Truong '{field_name}' phai co {expected_len} phan tu")

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

    sanitized = list(values)
    first_valid = next((v for v in sanitized if v > 0), None)
    if first_valid is None:
        return sanitized

    last_valid = first_valid
    for idx, value in enumerate(sanitized):
        if value > 0:
            last_valid = value
        else:
            sanitized[idx] = last_valid

    return sanitized


def _remove_dc(signal: np.ndarray, window_size: int) -> np.ndarray:
    if window_size < 2 or signal.size == 0:
        return signal - float(np.mean(signal))
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    baseline = np.convolve(signal, kernel, mode="same")
    return signal - baseline


def _bandpass_fft(signal: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(signal)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=signal.size)


def _estimate_bpm_from_ir(ir_values: List[int], sample_interval_ms: int) -> float | None:
    if len(ir_values) < 20:
        return None

    signal = np.array(ir_values, dtype=np.float64)
    signal_std = np.std(signal)
    
    # If signal has no variation, can't detect heartbeat
    if signal_std < 100:
        print(f"⚠️  IR signal co std = {signal_std:.2f} (qua thap, ko phat hien duoc nhip tim)")
        return None

    sample_rate_hz = _sample_rate_hz(sample_interval_ms)
    dc_window = max(2, int(sample_rate_hz * 1.0))
    centered = _remove_dc(signal, dc_window)
    filtered = _bandpass_fft(centered, sample_rate_hz, 0.5, 4.0)

    filtered_std = float(np.std(filtered))
    if filtered_std <= 1e-6:
        print("⚠️  IR filtered std qua thap, bo qua BPM")
        return None

    normalized = filtered / filtered_std
    threshold = max(0.4, 0.5 * float(np.std(normalized)))
    min_gap = max(1, int(sample_rate_hz * 0.40))

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

    if len(peaks) < 2:
        print(f"⚠️  Khong phat hien duoc peaks trong IR signal (found {len(peaks)} peaks)")
        return None

    rr_seconds = np.diff(peaks) / sample_rate_hz
    rr_seconds = rr_seconds[(rr_seconds > 0.25) & (rr_seconds < 2.0)]
    if rr_seconds.size == 0:
        print(f"⚠️  Peak intervals ko hop le, ko tinh duoc BPM")
        return None

    bpm = 60.0 / float(np.median(rr_seconds))
    return float(np.clip(bpm, MIN_BPM, MAX_BPM))


def _estimate_spo2_from_ir_red(
    ir_values: List[int],
    red_values: List[int],
    sample_interval_ms: int,
) -> float | None:
    """
    Estimate SpO2 using AC/DC method.
    
    Công thức:
      DC = mean(signal)
      AC = RMS(signal - DC) = sqrt(mean((signal - DC)²))
      SpO₂ = 110 - 25 × (AC_red/DC_red) / (AC_ir/DC_ir)
    """
    if len(ir_values) < 20 or len(red_values) < 20:
        return None

    ir = np.array(ir_values, dtype=np.float64)
    red = np.array(red_values, dtype=np.float64)

    sample_rate_hz = _sample_rate_hz(sample_interval_ms)
    dc_window = max(2, int(sample_rate_hz * 1.0))

    ir_dc = float(np.mean(ir))
    red_dc = float(np.mean(red))

    ir_centered = _remove_dc(ir, dc_window)
    red_centered = _remove_dc(red, dc_window)

    ir_filt = _bandpass_fft(ir_centered, sample_rate_hz, 0.5, 4.0)
    red_filt = _bandpass_fft(red_centered, sample_rate_hz, 0.5, 4.0)

    ir_ac = float(np.std(ir_filt))
    red_ac = float(np.std(red_filt))

    if ir_dc <= 1e-6 or red_dc <= 1e-6 or ir_ac <= 1e-6:
        return None

    # Ratio: (AC_red/DC_red) / (AC_ir/DC_ir)
    ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
    
    # SpO₂ công thức
    spo2 = 110.0 - 25.0 * ratio
    return float(np.clip(spo2, MIN_SPO2, MAX_SPO2))


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
# EMA (Exponential Moving Average): T_smooth = 0.3T + 0.7T_old

_last_temperature = 36.5
_temperature_lock = None  # Will be set by app.py if needed

def _median_filter_temperature(temps: List[float]) -> float:
    """
    Median filter: T = median(T_{i-2}, T_{i-1}, T_i)
    
    Input: List of temperatures (up to 3 values)
    Output: Median value (smoothed)
    """
    if not temps:
        return 36.5
    return float(np.median(np.array(temps, dtype=np.float64)))


def _ema_smooth_temperature(temp_new: float, temp_old: float = None, alpha: float = 0.3) -> float:
    """
    EMA (Exponential Moving Average): T_smooth = α×T_new + (1-α)×T_old
    
    Args:
        temp_new: Mẫu nhiệt độ mới từ MAX30205
        temp_old: Giá trị EMA trước đó (default: từ _last_temperature)
        alpha: Smooth factor (default: 0.3)
    
    Returns:
        Giá trị nhiệt độ đã làm mượt
    """
    global _last_temperature
    
    if temp_old is None:
        temp_old = _last_temperature
    
    temp_smooth = alpha * temp_new + (1 - alpha) * temp_old
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
            series_map[key] = _to_float_list(normalized[key], key)
        else:
            series_map[key] = [0.0] * sample_count

    for key in ("ax", "ay", "az", "gx", "gy", "gz"):
        if len(series_map[key]) != sample_count:
            raise ValueError(f"Cac truong mang phai cung do dai, loi tai '{key}'")

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

    temp_value = _to_float(normalized.get("temp", 0.0), "temp")

    clean_ir_series = _sanitize_optical_series(ir_series) if ir_series is not None else None
    clean_red_series = _sanitize_optical_series(red_series) if red_series is not None else None

    heart_rate: float | None = None
    if _is_valid_vital(normalized.get("heart_rate")):
        heart_rate = _to_float(normalized["heart_rate"], "heart_rate")
    elif clean_ir_series is not None:
        heart_rate = _estimate_bpm_from_ir(clean_ir_series, sample_interval_ms)

    if heart_rate is None:
        heart_rate = 0.0

    spo2_value: float | None = None
    if _is_valid_vital(normalized.get("spo2")):
        spo2_value = _to_float(normalized["spo2"], "spo2")
    elif clean_ir_series is not None and clean_red_series is not None:
        spo2_value = _estimate_spo2_from_ir_red(clean_ir_series, clean_red_series, sample_interval_ms)

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
                ir=clean_ir_series[idx] if clean_ir_series is not None else None,
                red=clean_red_series[idx] if clean_red_series is not None else None,
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
def classify(bpm: float, temp: float) -> str:
    """
    Classify health status based on BPM and body temperature.
    
    Thresholds (công thức):
    - Temperature: > 37.5°C (FEVER) | < 35°C (Hypothermia - nguy hiểm)
    - BPM: > 120 (HIGH) | < 50 (LOW)
    """
    # Kiểm tra nhiệt độ (ưu tiên)
    if temp > TEMP_FEVER_THRESHOLD:
        return STATUS_FEVER
    if temp < TEMP_DANGER_THRESHOLD:
        print(f"🚨 [HEALTH] HYPOTHERMIA: Temp = {temp:.1f}°C < {TEMP_DANGER_THRESHOLD}°C")
    
    # Kiểm tra nhịp tim
    if bpm > BPM_HIGH_THRESHOLD:
        return STATUS_HIGH_HEART_RATE
    if bpm < BPM_LOW_THRESHOLD:
        return STATUS_LOW_HEART_RATE
    
    return STATUS_NORMAL
