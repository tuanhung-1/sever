from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import PROJECT_ROOT, settings


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "fall_detection" / "multistage"
LEGACY_MODEL_FILE = "fall_cnn_gated_nolambda.keras"
V5_MODEL_FILE = "fall_v5_hybrid_deep.keras"
LEGACY_METADATA_FILE = "fall_cnn_metadata.json"
V5_METADATA_FILE = "fall_v5_metadata.json"
LEGACY_MEAN_FILE = "fall_cnn_mean.npy"
V5_MEAN_FILE = "fall_v5_sequence_mean.npy"
LEGACY_STD_FILE = "fall_cnn_std.npy"
V5_STD_FILE = "fall_v5_sequence_std.npy"
V5_TABULAR_FILE = "fall_v5_tabular.joblib"
V5_STATS_SCALER_FILE = "fall_v5_stats_scaler.joblib"
KERAS_CONFIG_DROP_KEYS = {
    "renorm",
    "renorm_clipping",
    "renorm_momentum",
    "quantization_config",
}
MIN_ALERT_CONFIDENCE = 0.075


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


def _load_joblib(path: Path):
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thieu joblib/scikit-learn de load tabular fall artifact. "
            "Cai dat bang: pip install scikit-learn joblib"
        ) from exc
    return joblib.load(path)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _metadata_threshold(metadata: dict[str, Any], fallback: float = 0.8) -> float:
    for key in ("threshold", "direct_alert_threshold", "candidate_threshold"):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    return fallback


def _sanitize_keras_config(obj: Any) -> int:
    removed = 0
    if isinstance(obj, dict):
        config = obj.get("config")
        if isinstance(config, dict):
            for key in KERAS_CONFIG_DROP_KEYS:
                if key in config:
                    config.pop(key, None)
                    removed += 1
        for value in obj.values():
            removed += _sanitize_keras_config(value)
    elif isinstance(obj, list):
        for value in obj:
            removed += _sanitize_keras_config(value)
    return removed


def _create_sanitized_keras_copy(model_path: Path) -> Path:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
    temp_path = Path(temp.name)
    temp.close()

    with zipfile.ZipFile(model_path, "r") as source:
        config = json.loads(source.read("config.json"))
        _sanitize_keras_config(config)

        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                if item.filename == "config.json":
                    data = json.dumps(config, separators=(",", ":")).encode("utf-8")
                else:
                    data = source.read(item.filename)
                target.writestr(item, data)

    return temp_path


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


class V5HybridFallModel(BaseFallModel):
    model_name = "fall_v5_hybrid_deep"

    def __init__(
        self,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        model_file: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.artifact_dir = _resolve_project_path(artifact_dir)
        self.metadata_path = self._resolve_artifact_file(
            V5_METADATA_FILE,
            LEGACY_METADATA_FILE,
        )
        self.metadata = _load_json(self.metadata_path)

        self.window_size = int(self.metadata.get("window_size", 200))
        self.feature_keys = list(
            self.metadata.get(
                "feature_keys",
                ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag"],
            )
        )
        self.handcrafted_feature_names = list(self.metadata.get("handcrafted_feature_names", []))
        self._model_file = model_file
        self._threshold = float(threshold if threshold is not None else _metadata_threshold(self.metadata))
        self._post_filter = dict(self.metadata.get("post_filter") or {})
        self._post_fall_confirmation = dict(self.metadata.get("post_fall_confirmation") or {})
        self._deep_weight = float((self.metadata.get("selected_spec") or {}).get("deep_weight", 1.0))
        self._deep_weight = min(max(self._deep_weight, 0.0), 1.0)

        self._mean = self._load_sequence_vector(V5_MEAN_FILE, LEGACY_MEAN_FILE)
        self._std = self._load_sequence_vector(V5_STD_FILE, LEGACY_STD_FILE)
        self._std = np.where(np.abs(self._std) < 1e-8, 1.0, self._std).astype(np.float32)

        self.model_path = self._resolve_model_path()
        tf = _import_tensorflow()
        self._model = self._load_model(tf)
        self._input_names = [tensor.name.split(":")[0] for tensor in getattr(self._model, "inputs", [])]
        self.tabular_path = self._resolve_optional_artifact_file(V5_TABULAR_FILE)
        self.stats_scaler_path = self._resolve_optional_artifact_file(V5_STATS_SCALER_FILE)
        self._tabular_model = self._load_tabular_model(self.tabular_path)

    def _load_model(self, tf):
        try:
            return tf.keras.models.load_model(str(self.model_path), compile=False)
        except TypeError:
            if self.model_path.suffix.lower() != ".keras":
                raise

            sanitized_path = _create_sanitized_keras_copy(self.model_path)
            try:
                return tf.keras.models.load_model(str(sanitized_path), compile=False)
            finally:
                try:
                    sanitized_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _resolve_artifact_file(self, *candidate_names: str) -> Path:
        for file_name in candidate_names:
            path = self.artifact_dir / file_name
            if path.exists():
                return path.resolve()
        joined = ", ".join(candidate_names)
        raise RuntimeError(f"Khong tim thay artifact trong {self.artifact_dir}: {joined}")

    def _resolve_optional_artifact_file(self, file_name: str) -> Path | None:
        path = self.artifact_dir / file_name
        return path.resolve() if path.exists() else None

    def _load_tabular_model(self, path: Path | None):
        if path is None:
            return None
        artifact = _load_joblib(path)
        if isinstance(artifact, dict):
            return artifact.get("model")
        return artifact

    def _load_sequence_vector(self, *candidate_names: str) -> np.ndarray:
        path = self._resolve_artifact_file(*candidate_names)
        values = np.load(path).astype(np.float32)
        if values.shape != (len(self.feature_keys),):
            raise RuntimeError(
                f"Scaler {path.name} co shape {values.shape}, "
                f"nhung sequence model can {(len(self.feature_keys),)}"
            )
        return values

    def _resolve_model_path(self) -> Path:
        requested = self._model_file
        candidates = []
        if requested:
            candidates.append(requested)
        candidates.extend([V5_MODEL_FILE, LEGACY_MODEL_FILE])

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if not path.is_absolute():
                path = self.artifact_dir / candidate if path.parent == Path(".") else PROJECT_ROOT / path
            if path.exists():
                return path.resolve()

        raise RuntimeError(
            "Khong tim thay file model fall AI. Can mot trong cac file: "
            f"{V5_MODEL_FILE}, {LEGACY_MODEL_FILE}"
        )

    def _trailing_mean(self, values: np.ndarray, window: int) -> np.ndarray:
        output = np.empty_like(values, dtype=np.float32)
        for idx in range(values.shape[0]):
            start = max(0, idx - window + 1)
            output[idx] = float(np.mean(values[start : idx + 1]))
        return output

    def _trailing_std(self, values: np.ndarray, window: int) -> np.ndarray:
        output = np.empty_like(values, dtype=np.float32)
        for idx in range(values.shape[0]):
            start = max(0, idx - window + 1)
            output[idx] = float(np.std(values[start : idx + 1]))
        return output

    def _trailing_max(self, values: np.ndarray, window: int) -> np.ndarray:
        output = np.empty_like(values, dtype=np.float32)
        for idx in range(values.shape[0]):
            start = max(0, idx - window + 1)
            output[idx] = float(np.max(values[start : idx + 1]))
        return output

    def _count_local_peaks(self, values: np.ndarray) -> float:
        if values.size < 3:
            return float(values.size)
        peaks = 0
        for idx in range(1, values.size - 1):
            if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
                peaks += 1
        return float(peaks)

    def _build_sequence_features(self, raw_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(raw_window, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] < 8:
            raise ValueError("raw_window phai co shape (samples, >=8)")

        raw = self._fit_window(raw[:, :8])
        ax, ay, az, gx, gy, gz, acc_mag_input, jerk = raw.T

        acc_mag = np.asarray(acc_mag_input, dtype=np.float32)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)
        acc_mag_diff = np.diff(acc_mag, prepend=acc_mag[:1]).astype(np.float32)
        gyro_mag_diff = np.diff(gyro_mag, prepend=gyro_mag[:1]).astype(np.float32)

        roll = np.degrees(np.arctan2(ay, az)).astype(np.float32)
        pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2))).astype(np.float32)

        acc_roll_mean = self._trailing_mean(acc_mag, 5)
        acc_roll_std = self._trailing_std(acc_mag, 5)
        acc_roll_max = self._trailing_max(acc_mag, 5)
        gyro_roll_std = self._trailing_std(gyro_mag, 5)
        acc_energy = (acc_mag**2).astype(np.float32)
        gyro_energy = (gyro_mag**2).astype(np.float32)

        sequence = np.column_stack([
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
            acc_mag,
            gyro_mag,
            acc_mag_diff,
            gyro_mag_diff,
            jerk,
            acc_roll_mean,
            acc_roll_std,
            acc_roll_max,
            gyro_roll_std,
            roll,
            pitch,
            acc_energy,
            gyro_energy,
        ]).astype(np.float32)

        peak_idx = int(np.argmax(acc_mag))
        handcrafted = self._build_handcrafted_features(sequence, peak_idx)
        return sequence, handcrafted

    def _build_handcrafted_features(self, sequence: np.ndarray, peak_idx: int) -> np.ndarray:
        values_by_name: dict[str, float] = {}
        for index, key in enumerate(self.feature_keys):
            column = sequence[:, index]
            values_by_name[f"{key}_mean"] = float(np.mean(column))
            values_by_name[f"{key}_std"] = float(np.std(column))
            values_by_name[f"{key}_min"] = float(np.min(column))
            values_by_name[f"{key}_max"] = float(np.max(column))
            values_by_name[f"{key}_range"] = float(np.max(column) - np.min(column))
            values_by_name[f"{key}_rms"] = float(np.sqrt(np.mean(column**2)))
            values_by_name[f"{key}_energy"] = float(np.sum(column**2))
            values_by_name[f"{key}_p95"] = float(np.percentile(column, 95))

        acc_mag = sequence[:, self.feature_keys.index("acc_mag")]
        gyro_mag = sequence[:, self.feature_keys.index("gyro_mag")]
        jerk = sequence[:, self.feature_keys.index("jerk")]
        roll = sequence[:, self.feature_keys.index("roll")]
        pitch = sequence[:, self.feature_keys.index("pitch")]

        post_slice = slice(min(max(peak_idx, 0), sequence.shape[0] - 1), None)
        pre_slice = slice(0, max(peak_idx, 1))

        values_by_name["acc_peak"] = float(np.max(acc_mag))
        values_by_name["acc_min"] = float(np.min(acc_mag))
        values_by_name["impact_range"] = float(np.max(acc_mag) - np.min(acc_mag))
        values_by_name["freefall_depth"] = float(max(0.0, 1.0 - np.min(acc_mag)))
        values_by_name["gyro_peak"] = float(np.max(gyro_mag))
        values_by_name["jerk_peak"] = float(np.max(jerk))
        values_by_name["acc_peak_count"] = self._count_local_peaks(acc_mag)
        values_by_name["gyro_peak_count"] = self._count_local_peaks(gyro_mag)
        values_by_name["jerk_peak_count"] = self._count_local_peaks(jerk)
        values_by_name["roll_change_deg"] = float(np.ptp(roll))
        values_by_name["pitch_change_deg"] = float(np.ptp(pitch))
        values_by_name["post_acc_std"] = float(np.std(acc_mag[post_slice]))
        values_by_name["post_gyro_mean"] = float(np.mean(gyro_mag[post_slice]))
        values_by_name["post_gyro_std"] = float(np.std(gyro_mag[post_slice]))
        values_by_name["post_jerk_mean"] = float(np.mean(jerk[post_slice]))
        values_by_name["post_acc_one_g_error"] = float(np.mean(np.abs(acc_mag[post_slice] - 1.0)))
        values_by_name["post_roll_std_deg"] = float(np.std(roll[post_slice]))
        values_by_name["post_pitch_std_deg"] = float(np.std(pitch[post_slice]))
        values_by_name["pre_acc_mean"] = float(np.mean(acc_mag[pre_slice]))
        values_by_name["post_acc_mean"] = float(np.mean(acc_mag[post_slice]))
        values_by_name["pre_gyro_mean"] = float(np.mean(gyro_mag[pre_slice]))
        values_by_name["post_gyro_mean_2"] = float(np.mean(gyro_mag[post_slice]))

        if len(self.handcrafted_feature_names) != 174:
            raise RuntimeError(
                f"metadata handcrafted_feature_names co {len(self.handcrafted_feature_names)} features, "
                "nhung model can 174"
            )

        return np.asarray([values_by_name.get(name, 0.0) for name in self.handcrafted_feature_names], dtype=np.float32)

    def _build_model_inputs(self, raw_window: np.ndarray):
        sequence, handcrafted = self._build_sequence_features(raw_window)
        standardized_sequence = self._standardize(sequence)
        sequence_batch = standardized_sequence[np.newaxis, :, :].astype(np.float32)
        handcrafted_batch = handcrafted[np.newaxis, :].astype(np.float32)

        if len(self._input_names) >= 2:
            return {
                self._input_names[0]: sequence_batch,
                self._input_names[1]: handcrafted_batch,
            }
        return [sequence_batch, handcrafted_batch]

    def _fit_window(self, raw_features: np.ndarray) -> np.ndarray:
        if raw_features.shape[0] == self.window_size:
            return raw_features
        if raw_features.shape[0] > self.window_size:
            return raw_features[: self.window_size]
        if raw_features.shape[0] == 0:
            return np.zeros((self.window_size, raw_features.shape[1]), dtype=np.float32)

        pad_count = self.window_size - raw_features.shape[0]
        pad = np.repeat(raw_features[-1:, :], pad_count, axis=0)
        return np.vstack([raw_features, pad]).astype(np.float32)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return ((features - self._mean) / self._std).astype(np.float32)

    def _predict_tabular_confidence(self, handcrafted: np.ndarray) -> float | None:
        if self._tabular_model is None:
            return None

        features = handcrafted[np.newaxis, :].astype(np.float32)
        if hasattr(self._tabular_model, "predict_proba"):
            probabilities = self._tabular_model.predict_proba(features)
            return float(np.asarray(probabilities).reshape(-1)[-1])

        prediction = self._tabular_model.predict(features)
        return float(np.asarray(prediction).reshape(-1)[0])

    def _ensemble_confidence(
        self,
        deep_confidence: float,
        tabular_confidence: float | None,
    ) -> float:
        if tabular_confidence is None:
            return deep_confidence
        return (
            self._deep_weight * deep_confidence
            + (1.0 - self._deep_weight) * tabular_confidence
        )

    def _strong_event_override(self, features: np.ndarray) -> dict[str, Any]:
        config = self._post_fall_confirmation
        acc_idx = self.feature_keys.index("acc_mag")
        gyro_idx = self.feature_keys.index("gyro_mag")
        jerk_idx = self.feature_keys.index("jerk")

        acc_peak = float(np.max(features[:, acc_idx]))
        gyro_peak = float(np.max(features[:, gyro_idx]))
        jerk_peak = float(np.max(features[:, jerk_idx]))

        min_acc = float(config.get("strong_acc_peak_g", 2.5))
        min_gyro = float(config.get("strong_gyro_peak_dps", 300.0))
        min_jerk = float(config.get("strong_jerk_peak_gps", 15.0))
        passed = acc_peak >= min_acc and gyro_peak >= min_gyro and jerk_peak >= min_jerk

        return {
            "enabled": bool(config),
            "passed": passed,
            "acc_peak_g": round(acc_peak, 4),
            "gyro_peak_dps": round(gyro_peak, 4),
            "jerk_peak_gps": round(jerk_peak, 4),
            "strong_acc_peak_g": min_acc,
            "strong_gyro_peak_dps": min_gyro,
            "strong_jerk_peak_gps": min_jerk,
        }

    def _post_filter_result(
        self,
        features: np.ndarray,
        confidence: float,
    ) -> tuple[bool, dict[str, Any]]:
        acc_idx = self.feature_keys.index("acc_mag")
        gyro_idx = self.feature_keys.index("gyro_mag")
        acc_peak = float(np.max(features[:, acc_idx]))
        gyro_peak = float(np.max(features[:, gyro_idx]))

        enabled = bool(self._post_filter.get("enabled", False))
        kind = str(self._post_filter.get("kind", "")).lower()
        min_acc = float(self._post_filter.get("min_acc_peak_g", 0.0))
        min_gyro = float(self._post_filter.get("min_gyro_peak_dps", 0.0))
        high_confidence_prob = float(self._post_filter.get("high_confidence_prob", 1.0))
        acc_passed = acc_peak >= min_acc
        gyro_passed = gyro_peak >= min_gyro
        high_confidence_passed = confidence >= high_confidence_prob

        if not enabled:
            passed = True
        elif kind == "soft_peak_filter":
            passed = acc_passed or gyro_passed
        else:
            passed = acc_passed and gyro_passed

        return passed, {
            "enabled": enabled,
            "kind": kind,
            "passed": passed,
            "acc_peak_g": round(acc_peak, 4),
            "gyro_peak_dps": round(gyro_peak, 4),
            "min_acc_peak_g": min_acc,
            "min_gyro_peak_dps": min_gyro,
            "high_confidence_prob": high_confidence_prob,
            "high_confidence_passed": high_confidence_passed,
            "acc_passed": acc_passed,
            "gyro_passed": gyro_passed,
        }

    def predict_raw_window(
        self,
        raw_window: np.ndarray,
        vitals: dict[str, float] | None = None,
    ) -> FallPrediction:
        model_inputs = self._build_model_inputs(raw_window)
        prediction = self._model.predict(model_inputs, verbose=0)
        deep_confidence = float(np.asarray(prediction).reshape(-1)[0])

        sequence, handcrafted = self._build_sequence_features(raw_window)
        tabular_confidence = self._predict_tabular_confidence(handcrafted)
        confidence = self._ensemble_confidence(deep_confidence, tabular_confidence)
        post_filter_passed, post_filter_details = self._post_filter_result(sequence, confidence)
        strong_override = self._strong_event_override(sequence)
        min_confidence_passed = confidence > MIN_ALERT_CONFIDENCE
        threshold_passed = confidence >= self._threshold
        detected = min_confidence_passed and (
            (threshold_passed and post_filter_passed) or bool(strong_override["passed"])
        )

        return FallPrediction(
            fall_detected=detected,
            confidence=confidence,
            details={
                "min_alert_confidence": MIN_ALERT_CONFIDENCE,
                "min_confidence_passed": min_confidence_passed,
                "threshold": round(self._threshold, 4),
                "threshold_passed": threshold_passed,
                "window_size": self.window_size,
                "feature_keys": self.feature_keys,
                "selected_predictor": self.metadata.get("selected_predictor"),
                "deep_confidence": round(deep_confidence, 6),
                "tabular_confidence": None if tabular_confidence is None else round(tabular_confidence, 6),
                "ensemble_confidence": round(confidence, 6),
                "deep_weight": round(self._deep_weight, 4),
                "model_path": str(self.model_path),
                "metadata_path": str(self.metadata_path),
                "artifact_dir": str(self.artifact_dir),
                "tabular_path": None if self.tabular_path is None else str(self.tabular_path),
                "stats_scaler_path": None if self.stats_scaler_path is None else str(self.stats_scaler_path),
                "sequence_mean_file": V5_MEAN_FILE,
                "sequence_std_file": V5_STD_FILE,
                "post_filter": post_filter_details,
                "strong_event_override": strong_override,
            },
        )


MultiStageGatedCnnFallModel = V5HybridFallModel


def create_fall_model() -> BaseFallModel:
    model = V5HybridFallModel(
        artifact_dir=settings.fall_model_dir,
        model_file=settings.fall_model_file,
        threshold=settings.fall_model_threshold,
    )
    print(
        "Loaded fall AI model: "
        f"{model.model_name} | artifacts={model.artifact_dir} | model={model.model_path}"
    )
    return model
