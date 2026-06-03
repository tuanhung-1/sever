from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts" / "fall_detection" / "multistage"


def _import_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thieu TensorFlow de load fall AI model. Cai dat bang: "
            "pip install -r requirements.txt"
        ) from exc
    return tf


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Khong tim thay metadata model: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class FallPrediction:
    fall_detected: bool
    confidence: float
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "detected": self.fall_detected,
            "confidence": round(self.confidence, 3),
        }


class BaseFallModel:
    model_name = "base"
    window_size = 0

    def predict_raw_window(
        self,
        raw6_window: np.ndarray,
        vitals: dict[str, float] | None = None,
    ) -> FallPrediction:
        raise NotImplementedError


class MultiStageGatedCnnFallModel(BaseFallModel):
    model_name = "multi_stage_gated_1d_cnn"

    def __init__(
        self,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        model_file: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self.metadata = _load_json(self.artifact_dir / "fall_cnn_metadata.json")

        self.window_size = int(self.metadata.get("window_size", 200))
        self.feature_keys = list(
            self.metadata.get(
                "feature_keys",
                ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag"],
            )
        )
        self._model_file = model_file
        self._threshold = float(threshold if threshold is not None else self.metadata.get("threshold", 0.9))
        self._post_filter = dict(self.metadata.get("post_filter") or {})

        self._mean = self._load_vector("fall_cnn_mean.npy")
        self._std = self._load_vector("fall_cnn_std.npy")
        self._std = np.where(np.abs(self._std) < 1e-8, 1.0, self._std).astype(np.float32)

        self.model_path = self._resolve_model_path()
        tf = _import_tensorflow()
        self._model = tf.keras.models.load_model(str(self.model_path), compile=False)

    def _load_vector(self, file_name: str) -> np.ndarray:
        path = self.artifact_dir / file_name
        if not path.exists():
            raise RuntimeError(f"Khong tim thay scaler artifact: {path}")
        values = np.load(path).astype(np.float32)
        if values.shape != (len(self.feature_keys),):
            raise RuntimeError(
                f"Scaler {file_name} co shape {values.shape}, "
                f"nhung model can {(len(self.feature_keys),)}"
            )
        return values

    def _resolve_model_path(self) -> Path:
        requested = self._model_file
        candidates = [requested] if requested else ["fall_cnn_gated_nolambda.keras"]

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if not path.is_absolute():
                path = self.artifact_dir / candidate
            if path.exists():
                return path.resolve()

        raise RuntimeError(
            "Khong tim thay file model fall AI. Can mot trong cac file: "
            "fall_cnn_gated_nolambda.keras"
        )

    def _raw6_to_features(self, raw6_window: np.ndarray) -> np.ndarray:
        raw6 = np.asarray(raw6_window, dtype=np.float32)
        if raw6.ndim != 2 or raw6.shape[1] < 6:
            raise ValueError("raw6_window phai co shape (samples, 6)")

        raw6 = self._fit_window(raw6[:, :6])
        ax, ay, az, gx, gy, gz = raw6.T
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        available = {
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "acc_mag": acc_mag,
            "gyro_mag": gyro_mag,
        }

        return np.column_stack([available[key] for key in self.feature_keys]).astype(np.float32)

    def _fit_window(self, raw6: np.ndarray) -> np.ndarray:
        if raw6.shape[0] == self.window_size:
            return raw6
        if raw6.shape[0] > self.window_size:
            return raw6[: self.window_size]
        if raw6.shape[0] == 0:
            return np.zeros((self.window_size, 6), dtype=np.float32)

        pad_count = self.window_size - raw6.shape[0]
        pad = np.repeat(raw6[-1:, :], pad_count, axis=0)
        return np.vstack([raw6, pad]).astype(np.float32)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return ((features - self._mean) / self._std).astype(np.float32)

    def _post_filter_result(self, features: np.ndarray) -> tuple[bool, dict[str, Any]]:
        acc_idx = self.feature_keys.index("acc_mag")
        gyro_idx = self.feature_keys.index("gyro_mag")
        acc_peak = float(np.max(features[:, acc_idx]))
        gyro_peak = float(np.max(features[:, gyro_idx]))

        enabled = bool(self._post_filter.get("enabled", False))
        min_acc = float(self._post_filter.get("min_acc_peak_g", 0.0))
        min_gyro = float(self._post_filter.get("min_gyro_peak_dps", 0.0))
        passed = (not enabled) or (acc_peak >= min_acc and gyro_peak >= min_gyro)

        return passed, {
            "enabled": enabled,
            "passed": passed,
            "acc_peak_g": round(acc_peak, 4),
            "gyro_peak_dps": round(gyro_peak, 4),
            "min_acc_peak_g": min_acc,
            "min_gyro_peak_dps": min_gyro,
        }

    def predict_raw_window(
        self,
        raw6_window: np.ndarray,
        vitals: dict[str, float] | None = None,
    ) -> FallPrediction:
        features = self._raw6_to_features(raw6_window)
        x = self._standardize(features)[np.newaxis, :, :]
        prediction = self._model.predict(x, verbose=0)
        confidence = float(np.asarray(prediction).reshape(-1)[0])

        post_filter_passed, post_filter_details = self._post_filter_result(features)
        detected = confidence >= self._threshold and post_filter_passed

        return FallPrediction(
            fall_detected=detected,
            confidence=confidence,
            details={
                "threshold": round(self._threshold, 4),
                "window_size": self.window_size,
                "feature_keys": self.feature_keys,
                "model_path": str(self.model_path),
                "post_filter": post_filter_details,
            },
        )


def create_fall_model() -> BaseFallModel:
    model = MultiStageGatedCnnFallModel(
        artifact_dir=settings.fall_model_dir,
        model_file=settings.fall_model_file,
        threshold=settings.fall_model_threshold,
    )
    print(f"Loaded fall AI model: {model.model_name} from {model.model_path}")
    return model
