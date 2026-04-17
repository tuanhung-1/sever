"""
model.py – Data model for incoming health telemetry payloads.

Expected payload format:
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
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

# ─── Thresholds ───────────────────────────────────────────────────────────────
TEMP_FEVER_THRESHOLD = 38.0   # °C
BPM_HIGH_THRESHOLD = 120.0    # beats per minute

# ─── Result labels ────────────────────────────────────────────────────────────
STATUS_FEVER = "FEVER"
STATUS_HIGH_HEART_RATE = "HIGH_HEART_RATE"
STATUS_NORMAL = "NORMAL"


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


def from_dict(payload: Dict[str, Any]) -> HealthData:
    """Parse and validate telemetry payload from a Python dict."""
    normalized = dict(payload)
    if "heart_rate" not in normalized and "bpm" in normalized:
        normalized["heart_rate"] = normalized["bpm"]
    if "timestamp" not in normalized and "ts" in normalized:
        normalized["timestamp"] = normalized["ts"]

    # Older device payloads may not include gyroscope/timestamp.
    normalized.setdefault("gx", 0.0)
    normalized.setdefault("gy", 0.0)
    normalized.setdefault("gz", 0.0)
    normalized.setdefault("timestamp", int(time.time()))
    normalized.setdefault("spo2", None)
    normalized.setdefault("ir", None)
    normalized.setdefault("red", None)

    required = ("ax", "ay", "az", "heart_rate", "temp")
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
    """Parse and validate telemetry payload from a JSON string."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Payload JSON khong hop le") from exc

    if not isinstance(payload, dict):
        raise ValueError("Payload khong hop le, can doi tuong JSON")

    return from_dict(payload)


def classify(bpm: float, temp: float) -> str:
    """Classify health status based on BPM and body temperature."""
    if temp > TEMP_FEVER_THRESHOLD:
        return STATUS_FEVER
    if bpm > BPM_HIGH_THRESHOLD:
        return STATUS_HIGH_HEART_RATE
    return STATUS_NORMAL
