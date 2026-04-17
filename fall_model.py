"""
fall_model.py - Fall detection model scaffold.

This module provides a small, stable interface so you can plug in a real AI
model later without changing the rest of the backend flow.
"""

from __future__ import annotations

import os
import importlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


AI_MODEL_NAME = "model.h5"
AI_MODEL_PATH = str(Path(__file__).resolve().parent / AI_MODEL_NAME)
AI_MEAN_PATH = str(Path(__file__).resolve().parent / "mean.npy")
AI_STD_PATH = str(Path(__file__).resolve().parent / "std.npy")
AI_THRESHOLD = float(os.getenv("AI_FALL_THRESHOLD", "0.6"))
AI_REQUIRE_FULL_WINDOW = os.getenv("AI_REQUIRE_FULL_WINDOW", "true").lower() == "true"
AI_ENABLE_EARLY_FALL = os.getenv("AI_ENABLE_EARLY_FALL", "true").lower() == "true"
AI_EARLY_ACCEL_THRESHOLD = float(os.getenv("AI_EARLY_ACCEL_THRESHOLD", "2.8"))
AI_EARLY_GYRO_THRESHOLD = float(os.getenv("AI_EARLY_GYRO_THRESHOLD", "4.8"))


def _resolve_effective_window(required_window: int) -> int:
    if AI_REQUIRE_FULL_WINDOW:
        return required_window

    raw = os.getenv("AI_INPUT_MAX_SAMPLES", str(required_window))
    try:
        requested = int(raw)
    except ValueError:
        requested = required_window

    requested = max(1, requested)
    return min(required_window, requested)


@dataclass(frozen=True)
class FallInput:
    """Input features for fall detection."""

    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    timestamp: int
    heart_rate: float = 0.0
    spo2: float = 0.0
    temp: float = 0.0


@dataclass(frozen=True)
class FallPrediction:
    """Standardized output from a fall detection model."""

    fall_detected: bool
    confidence: float
    label: str

    def to_dict(self) -> Dict[str, float | bool | str]:
        return {
            "fall_detected": self.fall_detected,
            "confidence": self.confidence,
            "label": self.label,
        }


class BaseFallModel:
    """Base interface for all fall detection implementations."""

    def predict(self, data: FallInput) -> FallPrediction:
        raise NotImplementedError("Can cai dat predict() trong model te nga cu the")


class RuleBasedFallModel(BaseFallModel):
    """
    Temporary model for testing pipeline.

    Replace this class with a real ML model later.
    Current logic uses simple thresholds from acceleration/gyro magnitude.
    """

    def __init__(self, accel_threshold: float = 2.6, gyro_threshold: float = 4.5) -> None:
        self.accel_threshold = accel_threshold
        self.gyro_threshold = gyro_threshold

    def predict(self, data: FallInput) -> FallPrediction:
        accel_mag = (data.ax ** 2 + data.ay ** 2 + data.az ** 2) ** 0.5
        gyro_mag = (data.gx ** 2 + data.gy ** 2 + data.gz ** 2) ** 0.5

        is_fall = accel_mag >= self.accel_threshold and gyro_mag >= self.gyro_threshold

        # Simple confidence proxy for test mode only.
        confidence = min(1.0, max(accel_mag / (self.accel_threshold * 1.5), gyro_mag / (self.gyro_threshold * 1.5)))

        return FallPrediction(
            fall_detected=is_fall,
            confidence=round(confidence, 3),
            label="FALL" if is_fall else "NO_FALL",
        )


class PlaceholderAIFallModel(BaseFallModel):
    """
    Placeholder for your real AI model.

    Expected flow (you will implement later):
    1) Load trained model (TensorFlow/PyTorch/ONNX/sklearn).
    2) Build feature vector from FallInput.
    3) Run inference.
    4) Map prediction to FallPrediction.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        try:
            tf = importlib.import_module("tensorflow")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Thieu TensorFlow. Cai dat bang: pip install tensorflow"
            ) from exc

        self._tf = tf
        self._model = tf.keras.models.load_model(model_path)
        self._mean = np.load(AI_MEAN_PATH).astype(np.float32)
        self._std = np.load(AI_STD_PATH).astype(np.float32)

        # Model was trained with shape (batch, timesteps, features).
        self._required_window = int(self._model.input_shape[1])
        self._feature_count = int(self._model.input_shape[2])
        self._effective_window = _resolve_effective_window(self._required_window)
        self._buffer: deque[np.ndarray] = deque(maxlen=self._effective_window)

        if self._feature_count != 11:
            raise RuntimeError(
                f"Model feature count khong hop le: {self._feature_count}. Can 11 features."
            )

    @staticmethod
    def _safe_float(value: float | None) -> float:
        if value is None:
            return 0.0
        return float(value)

    def _build_feature_vector(self, data: FallInput) -> np.ndarray:
        # Keep feature order aligned with training FEATURE_KEYS.
        roll = np.arctan2(data.ay, data.az + 1e-6)
        pitch = np.arctan2(-data.ax, np.sqrt(data.ay ** 2 + data.az ** 2) + 1e-6)

        vector = np.array(
            [
                data.ax,
                data.ay,
                data.az,
                data.gx,
                data.gy,
                data.gz,
                roll,
                pitch,
                data.heart_rate,
                self._safe_float(data.spo2),
                data.temp,
            ],
            dtype=np.float32,
        )
        return vector

    def _build_window_for_inference(self) -> np.ndarray:
        window = np.array(self._buffer, dtype=np.float32)

        # Optional fast mode: pad to model timesteps by repeating last row.
        if not AI_REQUIRE_FULL_WINDOW and window.shape[0] < self._required_window:
            pad_len = self._required_window - window.shape[0]
            pad_row = window[-1:]
            padding = np.repeat(pad_row, pad_len, axis=0)
            window = np.vstack([window, padding])

        window = (window - self._mean) / (self._std + 1e-6)
        return window.reshape(1, self._required_window, self._feature_count)

    @staticmethod
    def _early_fall_detected(data: FallInput) -> bool:
        accel_mag = float(np.sqrt(data.ax ** 2 + data.ay ** 2 + data.az ** 2))
        gyro_mag = float(np.sqrt(data.gx ** 2 + data.gy ** 2 + data.gz ** 2))
        return accel_mag >= AI_EARLY_ACCEL_THRESHOLD and gyro_mag >= AI_EARLY_GYRO_THRESHOLD

    def predict(self, data: FallInput) -> FallPrediction:
        feature_vec = self._build_feature_vector(data)
        self._buffer.append(feature_vec)

        if len(self._buffer) < self._effective_window:
            if AI_ENABLE_EARLY_FALL and self._early_fall_detected(data):
                return FallPrediction(
                    fall_detected=True,
                    confidence=0.99,
                    label="EARLY_FALL",
                )

            return FallPrediction(
                fall_detected=False,
                confidence=0.0,
                label="WARMUP",
            )

        window = self._build_window_for_inference()
        prob = float(self._model.predict(window, verbose=0)[0][0])
        is_fall = prob >= AI_THRESHOLD

        return FallPrediction(
            fall_detected=is_fall,
            confidence=round(prob, 3),
            label="FALL" if is_fall else "NO_FALL",
        )


def create_fall_model(use_ai: bool = False) -> BaseFallModel:
    """
    Factory for selecting model implementation.

    - use_ai=False: run test pipeline with RuleBasedFallModel.
    - use_ai=True : use your AI model at AI_MODEL_PATH.
    """
    if use_ai:
        return PlaceholderAIFallModel(model_path=AI_MODEL_PATH)
    return RuleBasedFallModel()


def build_fall_input_from_payload(payload: dict) -> FallInput:
    """Convert validated payload dict to FallInput."""
    return FallInput(
        ax=float(payload["ax"]),
        ay=float(payload["ay"]),
        az=float(payload["az"]),
        gx=float(payload["gx"]),
        gy=float(payload["gy"]),
        gz=float(payload["gz"]),
        timestamp=int(payload["timestamp"]),
    )
