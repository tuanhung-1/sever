"""
fall_model.py - Fall detection model scaffold.

This module provides a small, stable interface so you can plug in a real AI
model later without changing the rest of the backend flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


AI_MODEL_NAME = "model.h5"
AI_MODEL_PATH = str(Path(__file__).resolve().parent / AI_MODEL_NAME)


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

    def predict(self, data: FallInput) -> FallPrediction:

        raise NotImplementedError("Chua cai dat suy luan cho model AI te nga")


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
