from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _resolve_api_port() -> int:
    raw_port = os.getenv("API_PORT") or os.getenv("PORT") or "5050"
    try:
        return int(raw_port)
    except ValueError:
        return 5050


@dataclass(frozen=True)
class Settings:
    mqtt_broker: str
    mqtt_port: int
    mqtt_username: str
    mqtt_password: str
    mqtt_client_id: str
    mqtt_required: bool
    api_bind_host: str
    api_access_host: str
    api_port: int
    api_verbose_output: bool
    history_file: str
    fall_history_file: str
    fall_model_dir: str
    fall_model_file: str
    fall_model_threshold: float | None
    alert_cooldown_s: int
    alert_bpm_low: int
    alert_bpm_high: int
    alert_spo2_low: float
    alert_spo2_high: float
    alert_temp_low: float
    alert_temp_high: float
    alert_beep_on_ms: int
    alert_beep_off_ms: int
    alert_beep_duration_s: int
    sensor_input_process_delay_ms: int
    ws_health_emit_delay_ms: int
    ws_fall_emit_delay_ms: int
    fall_model_block_ms: int
    buzzer_beep_on_ms: int
    buzzer_beep_off_ms: int
    buzzer_beep_count: int
    buzzer_total_ms: int
    ppg_step_size: int

    @property
    def vital_thresholds(self) -> dict[str, dict[str, float]]:
        return {
            "bpm": {"low": self.alert_bpm_low, "high": self.alert_bpm_high},
            "spo2": {"low": self.alert_spo2_low, "high": self.alert_spo2_high},
            "temp": {"low": self.alert_temp_low, "high": self.alert_temp_high},
        }


settings = Settings(
    mqtt_broker=os.getenv(
        "MQTT_BROKER",
        "11060dbd13b54fc988ae8f9bfc43c089.s1.eu.hivemq.cloud",
    ),
    mqtt_port=_env_int("MQTT_PORT", 8883),
    mqtt_username=os.getenv("MQTT_USERNAME", "heart-rate"),
    mqtt_password=os.getenv("MQTT_PASSWORD", "aB123456"),
    mqtt_client_id=os.getenv("MQTT_CLIENT_ID", "python_backend1dsdssdaassdsasdd"),
    mqtt_required=_env_bool("MQTT_REQUIRED", False),
    api_bind_host=os.getenv("API_BIND_HOST", "0.0.0.0"),
    api_access_host=os.getenv("API_ACCESS_HOST", "192.168.1.23"),
    api_port=_resolve_api_port(),
    api_verbose_output=_env_bool("API_VERBOSE_OUTPUT", False),
    history_file=os.getenv("HISTORY_FILE", "storage/history/health_history.jsonl"),
    fall_history_file=os.getenv("FALL_HISTORY_FILE", "storage/history/fall_history.jsonl"),
    fall_model_dir=os.getenv(
        "FALL_MODEL_DIR",
        "artifacts/fall_detection/multistage",
    ),
    fall_model_file=os.getenv("FALL_MODEL_FILE", "fall_v5_hybrid_deep.keras"),
    fall_model_threshold=_env_optional_float("FALL_MODEL_THRESHOLD"),
    alert_cooldown_s=max(0, _env_int("ALERT_COOLDOWN_S", 20)),
    alert_bpm_low=_env_int("ALERT_BPM_LOW", 50),
    alert_bpm_high=_env_int("ALERT_BPM_HIGH", 120),
    alert_spo2_low=_env_float("ALERT_SPO2_LOW", 93.0),
    alert_spo2_high=_env_float("ALERT_SPO2_HIGH", 100.1),
    alert_temp_low=_env_float("ALERT_TEMP_LOW", 32.5),
    alert_temp_high=_env_float("ALERT_TEMP_HIGH", 39.5),
    alert_beep_on_ms=max(100, _env_int("ALERT_BEEP_ON_MS", 500)),
    alert_beep_off_ms=max(100, _env_int("ALERT_BEEP_OFF_MS", 700)),
    alert_beep_duration_s=max(10, _env_int("ALERT_BEEP_DURATION_S", 60)),
    sensor_input_process_delay_ms=max(0, _env_int("SENSOR_INPUT_PROCESS_DELAY_MS", 0)),
    ws_health_emit_delay_ms=max(0, _env_int("WS_HEALTH_EMIT_DELAY_MS", 0)),
    ws_fall_emit_delay_ms=max(0, _env_int("WS_FALL_EMIT_DELAY_MS", 0)),
    fall_model_block_ms=max(0, _env_int("FALL_MODEL_BLOCK_MS", 0)),
    buzzer_beep_on_ms=max(0, _env_int("BUZZER_BEEP_ON_MS", 2000)),
    buzzer_beep_off_ms=max(0, _env_int("BUZZER_BEEP_OFF_MS", 1000)),
    buzzer_beep_count=max(1, _env_int("BUZZER_BEEP_COUNT", 2)),
    buzzer_total_ms=max(0, _env_int("BUZZER_TOTAL_MS", 60000)),
    ppg_step_size=_env_int("PPG_STEP_SIZE", 80),
)
