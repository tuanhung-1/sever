"""
fall_model.py - Fall detection model implementations.

Supports GBDT model trained in gbdt/fall_model_output by default.
Optional LSTM/TensorFlow fallback can be enabled with FALL_MODEL_TYPE.
"""

from __future__ import annotations
import json
import os
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np



GBDT_DIR = Path(__file__).resolve().parent / "gbdt" / "fall_model_output"
GBDT_MODEL_PATH = str(GBDT_DIR / "fall_gbdt_model.joblib")
GBDT_METADATA_PATH = str(GBDT_DIR / "fall_metadata.json")

_STAT_KEYS = ["mean", "std", "min", "max", "range", "median", "skew", "kurt"]
_TIME_FEATURE_KEYS = [
    "ax", "ay", "az", "gx", "gy", "gz",
    "acc_mag", "gyro_mag", "jerk", "roll", "pitch",
]


def _import_gbdt_deps():
    try:
        import joblib
        from scipy.stats import skew, kurtosis
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thieu joblib/scikit-learn/scipy. Cai dat bang: "
            "pip install scikit-learn scipy joblib"
        ) from exc
    return joblib, skew, kurtosis


def _load_metadata(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class FallPrediction:
    """Standardized output from a fall detection model."""

    fall_detected: bool
    confidence: float
    details: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "detected": self.fall_detected,
            "confidence": round(self.confidence, 3),
        }


class BaseFallModel:
    """Base interface for all fall detection implementations."""

    model_name = "base"
    window_size = 0

    def predict_raw_window(
        self,
        raw6_window: np.ndarray,
        vitals: Dict[str, float] | None = None,
    ) -> FallPrediction:
        feats = self._extract_window_features(raw6_window)

        feature_vec = [
            float(feats.get(name, 0.0))
            for name in self._feature_names
        ]

        X = np.asarray([feature_vec], dtype=np.float32)

        prob = float(
            self._model.predict_proba(X)[:, 1][0]
        )

        detected = prob >= self._threshold

        return FallPrediction(
            fall_detected=detected,
            confidence=prob,
            details={
                "threshold": round(self._threshold, 4),
            },
        )


class GbdtFallModel(BaseFallModel):
    model_name = "gbdt"

    def __init__(self, model_path: str, metadata_path: str) -> None:
        joblib, skew, kurtosis = _import_gbdt_deps()
        metadata = _load_metadata(metadata_path)
        bundle = joblib.load(model_path)

        if isinstance(bundle, dict):
            self._model = bundle.get("model", bundle)
            feature_names = bundle.get("feature_names")
            threshold = bundle.get("threshold")
            window_size = bundle.get("window_size")
        else:
            self._model = bundle
            feature_names = None
            threshold = None
            window_size = None

        if not feature_names:
            feature_names = metadata.get("feature_names", [])
        if not feature_names:
            raise RuntimeError("Khong tim thay feature_names trong model metadata")

        self._feature_names = list(feature_names)
        self._threshold = float(threshold if threshold is not None else metadata.get("threshold", 0.5))
        self.window_size = int(window_size if window_size is not None else metadata.get("window_size", 150))
        self._skew = skew
        self._kurtosis = kurtosis

    def _safe_stat(self, x: np.ndarray) -> list[float]:
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return [0.0] * 8

        vals = [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.ptp(x)),
            float(np.median(x)),
        ]

        if x.size > 2:
            s = float(self._skew(x, bias=False))
            vals.append(s if np.isfinite(s) else 0.0)
        else:
            vals.append(0.0)

        if x.size > 3:
            k = float(self._kurtosis(x, bias=False))
            vals.append(k if np.isfinite(k) else 0.0)
        else:
            vals.append(0.0)

        return vals

    def _raw6_to_time_features(self, raw6: np.ndarray) -> np.ndarray:
        raw6 = np.asarray(raw6, dtype=np.float32)
        ax, ay, az, gx, gy, gz = raw6.T

        acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        gyro_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        jerk = np.abs(np.diff(acc_mag, prepend=acc_mag[0]))
        roll = np.arctan2(ay, az + 1e-6)
        pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2) + 1e-6)

        return np.column_stack([
            ax, ay, az, gx, gy, gz,
            acc_mag, gyro_mag, jerk, roll, pitch,
        ]).astype(np.float32)

    def _extract_window_features(self, raw6_window: np.ndarray) -> Dict[str, float]:
        tfm = self._raw6_to_time_features(raw6_window)
        d: Dict[str, float] = {}

        for i, name in enumerate(_TIME_FEATURE_KEYS):
            vals = self._safe_stat(tfm[:, i])
            for k, v in zip(_STAT_KEYS, vals):
                d[f"{name}_{k}"] = v

        acc = tfm[:, _TIME_FEATURE_KEYS.index("acc_mag")]
        gyro = tfm[:, _TIME_FEATURE_KEYS.index("gyro_mag")]
        jerk = tfm[:, _TIME_FEATURE_KEYS.index("jerk")]
        roll = tfm[:, _TIME_FEATURE_KEYS.index("roll")]
        pitch = tfm[:, _TIME_FEATURE_KEYS.index("pitch")]

        d["acc_peak"] = float(np.max(acc))
        d["acc_min"] = float(np.min(acc))
        d["freefall_depth"] = float(max(0.0, 1.0 - np.min(acc)))
        d["impact_minus_min"] = float(np.max(acc) - np.min(acc))
        d["jerk_peak"] = float(np.max(jerk))
        d["gyro_peak"] = float(np.max(gyro))

        d["roll_change_deg"] = float(np.ptp(roll) * 180 / np.pi)
        d["pitch_change_deg"] = float(np.ptp(pitch) * 180 / np.pi)
        d["angle_change_deg"] = float(max(d["roll_change_deg"], d["pitch_change_deg"]))
        d["roll_final_deg"] = float(roll[-1] * 180 / np.pi)
        d["pitch_final_deg"] = float(pitch[-1] * 180 / np.pi)

        tail = max(10, len(acc) // 3)
        d["post_acc_std"] = float(np.std(acc[-tail:]))
        d["post_gyro_std"] = float(np.std(gyro[-tail:]))
        d["post_motion_score"] = float(d["post_acc_std"] + 0.01 * d["post_gyro_std"])

        if np.std(acc) > 1e-6 and np.std(gyro) > 1e-6:
            corr = float(np.corrcoef(acc, gyro)[0, 1])
            d["acc_gyro_corr"] = corr if np.isfinite(corr) else 0.0
        else:
            d["acc_gyro_corr"] = 0.0

        return d

    def _physical_gate_from_raw6(self, raw6_window: np.ndarray) -> tuple[bool, Dict[str, float]]:
        tfm = self._raw6_to_time_features(raw6_window)
        acc = tfm[:, _TIME_FEATURE_KEYS.index("acc_mag")]
        gyro = tfm[:, _TIME_FEATURE_KEYS.index("gyro_mag")]
        jerk = tfm[:, _TIME_FEATURE_KEYS.index("jerk")]
        roll = tfm[:, _TIME_FEATURE_KEYS.index("roll")]
        pitch = tfm[:, _TIME_FEATURE_KEYS.index("pitch")]

        acc_peak = float(np.max(acc))
        acc_min = float(np.min(acc))
        jerk_peak = float(np.max(jerk))
        gyro_peak = float(np.max(gyro))
        angle_change_deg = float(max(np.ptp(roll), np.ptp(pitch)) * 180 / np.pi)
        post_acc_std = float(np.std(acc[-max(10, len(acc) // 3):]))

        info = {
            "acc_peak": round(acc_peak, 4),
            "acc_min": round(acc_min, 4),
            "jerk_peak": round(jerk_peak, 4),
            "gyro_peak": round(gyro_peak, 4),
            "angle_change_deg": round(angle_change_deg, 2),
            "post_acc_std": round(post_acc_std, 4),
        }

        suspicious = (
            (acc_peak >= 1.6 and jerk_peak >= 0.25) or
            (gyro_peak >= 180 and angle_change_deg >= 25) or
            (acc_min <= 0.55 and acc_peak >= 1.4)
        )

        return suspicious, info

    def predict_raw_window(
        self,
        raw6_window: np.ndarray,
        vitals: Dict[str, float] | None = None,
    ) -> FallPrediction:
        raw6_window = np.asarray(raw6_window, dtype=np.float32)
        suspicious, gate = self._physical_gate_from_raw6(raw6_window)

        if not suspicious:
            return FallPrediction(
                fall_detected=False,
                confidence=0.0,
                details={"gate": {**gate, "reason": "not_suspicious_physics"}},
            )

        feats = self._extract_window_features(raw6_window)
        feature_vec = [float(feats.get(name, 0.0)) for name in self._feature_names]
        X = np.asarray([feature_vec], dtype=np.float32)

        prob = float(self._model.predict_proba(X)[:, 1][0])
        detected = prob >= self._threshold

        return FallPrediction(
            fall_detected=detected,
            confidence=prob,
            details={
                "threshold": round(self._threshold, 4),
                "gate": {**gate, "reason": "run_model"},
            },
        )


def create_fall_model() -> BaseFallModel:
    

    if not os.path.exists(GBDT_MODEL_PATH):
        raise RuntimeError(
            f"❌ Không tìm thấy GBDT model tại: {GBDT_MODEL_PATH}"
        )

    model = GbdtFallModel(
        model_path=GBDT_MODEL_PATH,
        metadata_path=GBDT_METADATA_PATH,
    )

    print(f"✅ Loaded GBDT Fall Model from {GBDT_MODEL_PATH}")

    return model


# ─── Fallback Detection Methods (feature/add-fallback-detection) ──────────────
#
# def detect_fall_by_acceleration_threshold(ax: float, ay: float, az: float, 
#                                           threshold_g: float = 2.5) -> bool:
#     """
#     Simple fallback fall detection using acceleration magnitude threshold.
#     
#     Detects sudden vertical drops by checking total acceleration magnitude
#     against threshold. Useful when ML model is unavailable.
#     
#     Physics:
#     - During fall: acceleration magnitude spikes (free fall + impact)
#     - threshold_g: acceleration threshold in g-forces (default 2.5g)
#     
#     Args:
#         ax, ay, az: Accelerometer readings (m/s²)
#         threshold_g: Detection threshold in g-forces
#         
#     Returns:
#         True if acceleration exceeds threshold (likely fall)
#     """
#     g_force = 9.81  # m/s²
#     acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
#     threshold_ms2 = threshold_g * g_force
#     return acc_magnitude > threshold_ms2


# def detect_fall_by_gyro_spike(gx: float, gy: float, gz: float,
#                                threshold_dps: float = 500) -> bool:
#     """
#     Detect fall using gyroscope angular velocity spike.
#     
#     During a fall, body rotates rapidly. Large gyroscope spikes
#     indicate sudden rotational movement typical of falls.
#     
#     Args:
#         gx, gy, gz: Gyroscope readings (deg/s)
#         threshold_dps: Angular velocity threshold in deg/s
#         
#     Returns:
#         True if any axis exceeds threshold
#     """
#     gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
#     return gyro_magnitude > threshold_dps


# def ensemble_fall_detection(raw_imu: Dict[str, Any], 
#                             ml_model_available: bool,
#                             ml_confidence: float = 0.0) -> Dict[str, Any]:
#     """
#     Ensemble fall detection combining ML model + simple heuristics.
#     
#     Decision logic:
#     - If ML model available and confident (>0.75): trust ML
#     - If acceleration threshold exceeded: flag as potential fall
#     - If gyro spike + low heart rate: flag as fall
#     - Combine signals for robust detection
#     
#     Args:
#         raw_imu: IMU data dictionary with ax,ay,az,gx,gy,gz,heart_rate
#         ml_model_available: Whether ML model is ready
#         ml_confidence: ML model confidence score (0-1)
#         
#     Returns:
#         Dictionary with:
#         - fall_detected: bool
#         - confidence: float (0-1)
#         - method: str (ml|accel|gyro|ensemble)
#         - reasons: List[str]
#     """
#     result = {
#         "fall_detected": False,
#         "confidence": 0.0,
#         "method": "none",
#         "reasons": []
#     }
#     
#     if not raw_imu:
#         return result
#     
#     # Try ML model first if available
#     if ml_model_available and ml_confidence > 0.75:
#         result["fall_detected"] = True
#         result["confidence"] = ml_confidence
#         result["method"] = "ml"
#         result["reasons"].append(f"ML model confidence: {ml_confidence:.2%}")
#         return result
#     
#     # Fallback to heuristics
#     accel_spike = detect_fall_by_acceleration_threshold(
#         raw_imu.get("ax", 0), raw_imu.get("ay", 0), raw_imu.get("az", 0)
#     )
#     gyro_spike = detect_fall_by_gyro_spike(
#         raw_imu.get("gx", 0), raw_imu.get("gy", 0), raw_imu.get("gz", 0)
#     )
#     
#     if accel_spike and gyro_spike:
#         result["fall_detected"] = True
#         result["confidence"] = 0.85
#         result["method"] = "ensemble"
#         result["reasons"].extend([
#             "High acceleration magnitude",
#             "High angular velocity"
#         ])
#     elif accel_spike:
#         result["fall_detected"] = True
#         result["confidence"] = 0.70
#         result["method"] = "accel"
#         result["reasons"].append("Acceleration threshold exceeded")
#     
#     return result