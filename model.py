"""
model.py – Data model for incoming health telemetry payloads.

Supported payload formats:

1) Single-sample payload:
{
    "ts": 1710000000,
    "bpm": 78,
    "spo2": 97,
    "temp": 36.5,
  "ax": 0.12,
  "ay": -0.98,
  "az": 9.81,
  "gx": 0.01,
  "gy": 0.02,
  "gz": 0.00,
    "ir": 523456,
    "red": 498123
}

2) Batch payload (from edge node):
{
    "ts": 1710000000,
    "temp": 36.5,
    "sample_interval_ms": 20,
    "ir": [523456, 523500, ...],
    "red": [498123, 498200, ...],
    "ax": [0.12, 0.15, ...],
    "ay": [-0.98, -1.02, ...],
    "az": [9.81, 9.76, ...],
    "gx": [0.01, 0.03, ...],
    "gy": [0.02, 0.01, ...],
    "gz": [0.00, -0.01, ...]
}
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

# ─── Thresholds ───────────────────────────────────────────────────────────────
TEMP_FEVER_THRESHOLD = 38.0   # °C
BPM_HIGH_THRESHOLD = 120.0    # beats per minute

# ─── Result labels ────────────────────────────────────────────────────────────
STATUS_FEVER = "FEVER"
STATUS_HIGH_HEART_RATE = "HIGH_HEART_RATE"
STATUS_NORMAL = "NORMAL"

# ─── Batch parsing defaults ───────────────────────────────────────────────────
DEFAULT_BATCH_SAMPLE_INTERVAL_MS = int(os.getenv("BATCH_SAMPLE_INTERVAL_MS", "20"))
MIN_BPM = 35.0
MAX_BPM = 220.0
MIN_SPO2 = 70.0
MAX_SPO2 = 100.0


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

        # Preferred device keys.
        payload["bpm"] = int(round(payload["heart_rate"]))
        payload["ts"] = payload["timestamp"]

        # Keep backward-compatible aliases for existing clients.
        payload["timestamp"] = payload["timestamp"]
        payload["heart_rate"] = payload["heart_rate"]
        return payload


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


def _is_valid_vital(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(value) >= 0.0
    except (TypeError, ValueError):
        return False


def _estimate_bpm_from_ir(ir_values: List[int], sample_interval_ms: int) -> float | None:
    if len(ir_values) < 20:
        return None

    signal = np.array(ir_values, dtype=np.float64)
    if np.std(signal) < 1e-6:
        return None

    # Light smoothing to suppress sensor noise.
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(signal, kernel, mode="same")
    centered = smooth - np.mean(smooth)

    threshold = max(1.0, 0.45 * np.std(centered))
    sample_rate_hz = 1000.0 / float(max(1, sample_interval_ms))
    min_gap = max(1, int(sample_rate_hz * 0.30))

    peaks: List[int] = []
    for idx in range(1, len(centered) - 1):
        is_peak = centered[idx] > centered[idx - 1] and centered[idx] >= centered[idx + 1]
        if not is_peak or centered[idx] <= threshold:
            continue

        if not peaks or idx - peaks[-1] >= min_gap:
            peaks.append(idx)
            continue

        if centered[idx] > centered[peaks[-1]]:
            peaks[-1] = idx

    if len(peaks) < 2:
        return None

    rr_seconds = np.diff(peaks) / sample_rate_hz
    rr_seconds = rr_seconds[(rr_seconds > 0.25) & (rr_seconds < 2.0)]
    if rr_seconds.size == 0:
        return None

    bpm = 60.0 / float(np.median(rr_seconds))
    return float(np.clip(bpm, MIN_BPM, MAX_BPM))


def _estimate_spo2_from_ir_red(ir_values: List[int], red_values: List[int]) -> float | None:
    if len(ir_values) < 20 or len(red_values) < 20:
        return None

    ir = np.array(ir_values, dtype=np.float64)
    red = np.array(red_values, dtype=np.float64)

    ir_dc = float(np.mean(ir))
    red_dc = float(np.mean(red))
    ir_ac = float(np.std(ir))
    red_ac = float(np.std(red))

    if ir_dc <= 1e-6 or red_dc <= 1e-6 or ir_ac <= 1e-6:
        return None

    ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
    spo2 = 110.0 - 25.0 * ratio
    return float(np.clip(spo2, MIN_SPO2, MAX_SPO2))


def _is_batch_payload(payload: Dict[str, Any]) -> bool:
    series_keys = ("ax", "ay", "az", "gx", "gy", "gz")
    return any(isinstance(payload.get(key), list) for key in series_keys)


def _normalize_payload_aliases(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)

    if "heart_rate" not in normalized and "bpm" in normalized:
        normalized["heart_rate"] = normalized["bpm"]
    if "timestamp" not in normalized and "ts" in normalized:
        normalized["timestamp"] = normalized["ts"]

    return normalized


def from_batch_dict(payload: Dict[str, Any]) -> List[HealthData]:
    """Parse batch telemetry payload into a list of normalized samples."""
    normalized = _normalize_payload_aliases(payload)

    required_series = ("ax", "ay", "az", "gx", "gy", "gz")
    series_map: Dict[str, List[float]] = {}
    for key in required_series:
        if key not in normalized:
            raise ValueError(f"Thieu truong bat buoc: {key}")
        series_map[key] = _to_float_list(normalized[key], key)

    sample_count = len(series_map["ax"])
    for key in required_series:
        if len(series_map[key]) != sample_count:
            raise ValueError(f"Cac truong mang phai cung do dai, loi tai '{key}'")

    ir_series = _to_optional_int_list(normalized.get("ir"), "ir", sample_count)
    red_series = _to_optional_int_list(normalized.get("red"), "red", sample_count)

    sample_interval_ms = _to_int(
        normalized.get("sample_interval_ms", DEFAULT_BATCH_SAMPLE_INTERVAL_MS),
        "sample_interval_ms",
    )
    sample_interval_ms = max(1, sample_interval_ms)

    batch_end_timestamp = _to_int(
        normalized.get("timestamp", int(time.time() * 1000)),
        "timestamp",
    )
    batch_start_timestamp = batch_end_timestamp - (sample_count - 1) * sample_interval_ms

    temp_value = _to_float(normalized.get("temp", 0.0), "temp")

    heart_rate: float | None = None
    if _is_valid_vital(normalized.get("heart_rate")):
        heart_rate = _to_float(normalized["heart_rate"], "heart_rate")
    elif ir_series is not None:
        heart_rate = _estimate_bpm_from_ir(ir_series, sample_interval_ms)

    if heart_rate is None:
        heart_rate = 0.0

    spo2_value: float | None = None
    if _is_valid_vital(normalized.get("spo2")):
        spo2_value = _to_float(normalized["spo2"], "spo2")
    elif ir_series is not None and red_series is not None:
        spo2_value = _estimate_spo2_from_ir_red(ir_series, red_series)

    samples: List[HealthData] = []
    for idx in range(sample_count):
        sample_timestamp = batch_start_timestamp + idx * sample_interval_ms
        samples.append(
            HealthData(
                ax=series_map["ax"][idx],
                ay=series_map["ay"][idx],
                az=series_map["az"][idx],
                gx=series_map["gx"][idx],
                gy=series_map["gy"][idx],
                gz=series_map["gz"][idx],
                heart_rate=heart_rate,
                spo2=spo2_value,
                temp=temp_value,
                ir=ir_series[idx] if ir_series is not None else None,
                red=red_series[idx] if red_series is not None else None,
                timestamp=sample_timestamp,
            )
        )

    return samples


def from_dict(payload: Dict[str, Any]) -> HealthData:
    """Parse and validate telemetry payload from a Python dict."""
    normalized = _normalize_payload_aliases(payload)

    # Older device payloads may not include gyroscope/timestamp.
    normalized.setdefault("gx", 0.0)
    normalized.setdefault("gy", 0.0)
    normalized.setdefault("gz", 0.0)
    normalized.setdefault("timestamp", int(time.time()))
    normalized.setdefault("heart_rate", 0.0)
    normalized.setdefault("spo2", None)
    normalized.setdefault("ir", None)
    normalized.setdefault("red", None)

    required = ("ax", "ay", "az", "temp")
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ValueError(f"Thieu truong bat buoc: {', '.join(missing)}")

    return HealthData(
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


def classify(bpm: float, temp: float) -> str:
    """Classify health status based on BPM and body temperature."""
    if temp > TEMP_FEVER_THRESHOLD:
        return STATUS_FEVER
    if bpm > BPM_HIGH_THRESHOLD:
        return STATUS_HIGH_HEART_RATE
    return STATUS_NORMAL
