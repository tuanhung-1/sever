try:
    import paho.mqtt.client as mqtt
except ModuleNotFoundError:
    mqtt = None
import csv
import io
import ssl
import certifi
import time
import json
import traceback
import struct
from threading import Lock, Timer
from collections import deque

import numpy as np

from flask import Flask, Response, jsonify, request

from app.core.config import settings
from app.core.extensions import socketio
from app.models.fall import create_fall_model
from app.models.health import from_json_samples, classify, STATUS_FALL_DETECTED, estimate_bpm_from_ppg_values
from app.repositories.history_repository import append_jsonl_record, read_jsonl_records

_buzzer_active = False
_wait_next_normal_batch = False
# Lưu lịch sử nhiều batch
# Buffer lưu 3 batch gần nhất
_vital_batch_buffer = deque(maxlen=3)

# Cooldown tránh spam
_last_alert_time = 0
ALERT_COOLDOWN_S = settings.alert_cooldown_s

_batch_alert_lock = Lock()


VITAL_THRESHOLDS = settings.vital_thresholds
 
# Tần số beep: ON ms, OFF ms, số lần lặp trong 60 giây
ALERT_BEEP_ON_MS = settings.alert_beep_on_ms
ALERT_BEEP_OFF_MS = settings.alert_beep_off_ms
ALERT_BEEP_DURATION_S = settings.alert_beep_duration_s
 
# Danh sách timer đang chạy (để cancel khi có alert mới)
_alert_timers: list[Timer] = []
_alert_timers_lock = Lock()

BROKER = settings.mqtt_broker
MQTT_PORT = settings.mqtt_port
USERNAME = settings.mqtt_username
PASSWORD = settings.mqtt_password
CLIENT_ID = settings.mqtt_client_id
MQTT_REQUIRED = settings.mqtt_required
API_BIND_HOST = settings.api_bind_host
API_ACCESS_HOST = settings.api_access_host
HISTORY_FILE = settings.history_file
FALL_HISTORY_FILE = settings.fall_history_file
TRAINING_FALL_RAW_FILE = "storage/training/fall_raw_windows.jsonl"
TRAINING_FALL_RAW_CSV_FILE = "storage/training/fall_raw_training_windows.csv"
FALL_TRAINING_EXPORT_ENABLED = True
VALID_TRAINING_LABELS = {"fall", "not_fall"}
API_VERBOSE_OUTPUT = settings.api_verbose_output


API_PORT = settings.api_port
SENSOR_INPUT_PROCESS_DELAY_MS = settings.sensor_input_process_delay_ms
WS_HEALTH_EMIT_DELAY_MS = settings.ws_health_emit_delay_ms
WS_FALL_EMIT_DELAY_MS = settings.ws_fall_emit_delay_ms
FALL_MODEL_BLOCK_MS = settings.fall_model_block_ms
_PPG_STEP_SIZE = settings.ppg_step_size


def moving_average(arr, window_size):
    result = []
    for i in range(len(arr)):
        sumv = 0
        count = 0
        for j in range(i - window_size, i + window_size + 1):
            if 0 <= j < len(arr):
                sumv += arr[j]
                count += 1
        result.append(sumv / count if count > 0 else arr[i])
    return result

def calculate_bpm_from_samples(samples):
    if len(samples) < 80:
        return None

    ir = [x['ir'] for x in samples]
    t = [x['t'] for x in samples]
    intervals = [
        t[i] - t[i - 1]
        for i in range(1, len(t))
        if isinstance(t[i], (int, float)) and isinstance(t[i - 1], (int, float)) and t[i] > t[i - 1]
    ]
    if intervals:
        sample_interval_ms = int(round(float(np.median(intervals))))
        bpm = estimate_bpm_from_ppg_values(ir, sample_interval_ms)
        if bpm is not None:
            return round(float(bpm), 1)

    filtered = moving_average(ir, 2)

    mean = sum(filtered) / len(filtered)

    peaks = []

    for i in range(1, len(filtered) - 1):

        if (
            filtered[i] > filtered[i - 1]
            and filtered[i] > filtered[i + 1]
            and filtered[i] > mean
        ):

            if not peaks or t[i] - peaks[-1]['t'] > 350:
                peaks.append({
                    'index': i,
                    't': t[i],
                    'value': filtered[i]
                })

    if len(peaks) < 2:
        return None

    intervals = []

    for i in range(1, len(peaks)):
        dt = peaks[i]['t'] - peaks[i - 1]['t']

        if 400 <= dt <= 1500:
            intervals.append(dt)

    if len(intervals) == 0:
        return None

    median_interval = sorted(intervals)[len(intervals) // 2]

    bpm = 60000 / median_interval

    if bpm < 40 or bpm > 180:
        return None

    return round(bpm, 1)



app = Flask(__name__)
socketio.init_app(app, cors_allowed_origins="*")


def create_app() -> Flask:
    return app

_latest_health_packet = None   # Chỉ chứa vitals (health_update)
_latest_fall_packet = None     # Chỉ chứa fall result (fall_update)
_latest_raw_payload = None
_packet_lock = Lock()
_history_lock = Lock()
_fall_model_lock = Lock()
_fall_model = None
_mqtt_client = None
_mqtt_client_lock = Lock()
_mqtt_connected = False
_current_temperature = 36.5
_temp_lock = Lock()

# Throttle timers riêng cho từng luồng
_health_emit_timer = None
_fall_emit_timer = None
_health_emit_timer_lock = Lock()
_fall_emit_timer_lock = Lock()


# ── Lưu latest vitals từ sensor/data ────────────────────────────────────────
_latest_hr = 0.0
_latest_spo2 = 0.0
_latest_vitals_lock = Lock()

# Last valid vitals preserved across packets
_last_valid_vitals = {"heart_rate": None, "spo2": None, "temp": None}
_last_valid_vital_times = {"heart_rate": None, "spo2": None, "temp": None}
_recent_bpm_values = deque(maxlen=7)
_recent_spo2_values = deque(maxlen=5)
_recent_temp_values = deque(maxlen=5)
VITAL_STALE_AFTER_S = 12.0

# ── Smooth BPM/SpO2 với EMA ──────────────────────────────────────────────

_ema_bpm = None
_ema_spo2 = None

# Alpha càng thấp -> càng mượt nhưng phản hồi chậm
EMA_ALPHA_BPM = 0.08
EMA_ALPHA_SPO2 = 0.10

_ema_lock = Lock()

def _is_valid_vital_value(key: str, value):
    try:
        if value is None:
            return False
        if key == "heart_rate":
            v = float(value)
            return 35.0 <= v <= 220.0
        if key == "spo2":
            v = float(value)
            return 70.0 <= v <= 100.0
        if key == "temp":
            v = float(value)
            return 25.0 <= v <= 45.0
    except Exception:
        return False
    return False


def _apply_ema(current_value, ema_value, alpha):
    """
    EMA smoothing:
        ema = alpha * current + (1-alpha) * previous
    """

    if current_value is None:
        return ema_value

    if ema_value is None:
        return float(current_value)

    return (
        alpha * float(current_value)
        + (1.0 - alpha) * float(ema_value)
    )


def _clamp(value, low, high):
    return max(low, min(high, value))


def _quality_is_valid(quality: dict) -> bool:
    if not isinstance(quality, dict):
        return True
    valid = quality.get("valid", True)
    if isinstance(valid, str):
        return valid.strip().lower() not in {"0", "false", "no", "off"}
    return bool(valid)


def _quality_confidence(quality: dict) -> float:
    if not _quality_is_valid(quality):
        return 0.0
    if not isinstance(quality, dict):
        return 0.75

    confidence = 1.0

    motion = _to_number(quality.get("bad_motion_rate"))
    if motion is not None:
        motion = _clamp(float(motion), 0.0, 0.25)
        confidence *= _clamp(1.0 - motion / 0.25 * 0.65, 0.35, 1.0)

    max_jerk = _to_number(quality.get("max_jerk"))
    if max_jerk is not None and max_jerk > 8.0:
        confidence *= _clamp(1.0 - (float(max_jerk) - 8.0) / 20.0 * 0.55, 0.45, 1.0)

    for key in ("ir_acdc", "red_acdc"):
        acdc = _to_number(quality.get(key))
        if acdc is None:
            continue
        if acdc < 0.004 or acdc > 0.18:
            confidence *= 0.75

    status = str(quality.get("status", "GOOD")).upper()
    if status not in {"GOOD", "UNKNOWN"}:
        confidence *= 0.60

    return round(_clamp(confidence, 0.0, 1.0), 3)


def _ppg_signal_is_usable(quality: dict) -> bool:
    if not _quality_is_valid(quality):
        return False
    if not isinstance(quality, dict):
        return True

    ir_p2p = _to_number(quality.get("ir_p2p"))
    red_p2p = _to_number(quality.get("red_p2p"))
    ir_acdc = _to_number(quality.get("ir_acdc"))
    red_acdc = _to_number(quality.get("red_acdc"))
    motion = _to_number(quality.get("bad_motion_rate"))

    if ir_p2p is not None and ir_p2p < 700:
        return False
    if red_p2p is not None and red_p2p < 300:
        return False
    if ir_acdc is not None and ir_acdc < 0.0035:
        return False
    if red_acdc is not None and red_acdc < 0.0035:
        return False
    if motion is not None and motion > 0.20:
        return False

    return True


def _last_average(values) -> float | None:
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return None
    ordered = sorted(numeric)
    if len(ordered) >= 5:
        ordered = ordered[1:-1]
    return sum(ordered) / len(ordered)


def _adaptive_alpha(base_alpha: float, confidence: float) -> float:
    confidence = _clamp(float(confidence), 0.0, 1.0)
    return _clamp(
        base_alpha * (0.60 + confidence),
        base_alpha * 0.50,
        min(0.60, base_alpha * 1.80),
    )


def _get_recent_valid_value(key: str, now_mono: float):
    value = _last_valid_vitals.get(key)
    updated_at = _last_valid_vital_times.get(key)
    if value is None:
        return None, True
    if updated_at is None:
        return value, False
    stale = (now_mono - updated_at) > VITAL_STALE_AFTER_S
    return (None if stale else value), stale


def _smooth_vital_value(
    key: str,
    raw_value,
    ema_value,
    recent_values,
    base_alpha: float,
    confidence: float,
    jump_threshold: float,
    max_delta_per_update: float | None,
    decimals: int,
    now_mono: float,
):
    if not _is_valid_vital_value(key, raw_value):
        fallback, stale = _get_recent_valid_value(key, now_mono)
        return fallback, ema_value, stale, False, None

    raw = round(float(raw_value), decimals)
    recent_values.append(raw)
    averaged = _last_average(recent_values)
    stable_input = raw if averaged is None else round(averaged, decimals)

    alpha = _adaptive_alpha(base_alpha, confidence)
    if ema_value is not None:
        jump = abs(raw - float(ema_value))
        if jump >= jump_threshold:
            alpha = min(alpha, base_alpha * 0.75)
        if key == "heart_rate" and jump >= 35.0 and raw < float(ema_value):
            alpha = max(alpha, 0.30)
            if max_delta_per_update is not None:
                max_delta_per_update = max(max_delta_per_update, 12.0)

    smoothed = _apply_ema(stable_input, ema_value, alpha)
    if ema_value is not None and max_delta_per_update is not None:
        smoothed = _clamp(
            smoothed,
            float(ema_value) - max_delta_per_update,
            float(ema_value) + max_delta_per_update,
        )
    return round(smoothed, decimals), smoothed, False, True, stable_input


def _choose_bpm_estimate(primary_bpm, fallback_bpm):
    primary_ok = _is_valid_vital_value("heart_rate", primary_bpm)
    fallback_ok = _is_valid_vital_value("heart_rate", fallback_bpm)

    if primary_ok and fallback_ok:
        primary = float(primary_bpm)
        fallback = float(fallback_bpm)
        diff = abs(primary - fallback)
        if diff >= 25.0:
            return fallback, "fallback_disagreement"
        if primary >= 130.0 and fallback <= 110.0:
            return fallback, "fallback_harmonic_guard"
        return primary, "primary"

    if primary_ok:
        return float(primary_bpm), "primary_only"
    if fallback_ok:
        return float(fallback_bpm), "fallback_only"
    return None, "missing"
# ── Trạng thái luồng Fall_Raw ────────────────────────────────────────────────
_waiting_for_fall_raw = False
_fall_raw_state_lock = Lock()
_pending_training_label = None
_pending_training_label_lock = Lock()


# ════════════════════════════════════════════════════════════════════════════════
# EMIT HELPERS — tách riêng health và fall
# ════════════════════════════════════════════════════════════════════════════════

def _emit_health(data: dict, delay_ms: int = WS_HEALTH_EMIT_DELAY_MS):
    """
    Emit Socket.IO event 'health_update'.
    Payload shape:
        {
            "type": "health_update",
            "source_topic": "sensor/data",
            "server_timestamp": <unix_s>,
            "data": {
                "ts": ...,
                "bpm": ...,
                "spo2": ...,
                "temp": ...,
                "status": [...]
            }
        }
    """
    global _health_emit_timer

    def do_emit():
        try:
            socketio.emit("health_update", data)
            print(f"📡 [EMIT] health_update after {delay_ms}ms")
        except Exception as e:
            print(f"❌ health_update emit error: {e}")

    with _health_emit_timer_lock:
        if _health_emit_timer is not None:
            _health_emit_timer.cancel()

        if delay_ms <= 0:
            do_emit()
            _health_emit_timer = None
            return

        _health_emit_timer = Timer(delay_ms / 1000.0, do_emit)
        _health_emit_timer.daemon = True
        _health_emit_timer.start()


def _emit_fall(data: dict, delay_ms: int = WS_FALL_EMIT_DELAY_MS):
    """
    Emit Socket.IO event 'fall_update'.
    Payload shape:
        {
            "type": "fall_update",
            "source_topic": "sensor/fall_raw",
            "server_timestamp": <unix_s>,
            "fall": {
                "detected": bool,
                "confidence": float
            }
        }
    """
    global _fall_emit_timer

    def do_emit():
        try:
            socketio.emit("fall_update", data)
            print(f"📡 [EMIT] fall_update after {delay_ms}ms")
        except Exception as e:
            print(f"❌ fall_update emit error: {e}")

    with _fall_emit_timer_lock:
        if _fall_emit_timer is not None:
            _fall_emit_timer.cancel()

        if delay_ms <= 0:
            do_emit()
            _fall_emit_timer = None
            return

        _fall_emit_timer = Timer(delay_ms / 1000.0, do_emit)
        _fall_emit_timer.daemon = True
        _fall_emit_timer.start()


# ════════════════════════════════════════════════════════════════════════════════
# PACKET BUILDERS
# ════════════════════════════════════════════════════════════════════════════════

def _build_health_packet(health_data, source_topic: str) -> dict:
    """
    Build health_update packet — chỉ chứa vitals, KHÔNG chứa fall.
    """
    raw_data = health_data.to_dict()
    data = _normalize_health_data_for_frontend(raw_data)

    packet = {
        "type": "health_update",
        "source_topic": source_topic,
        "server_timestamp": int(time.time()),
        "data": data,
    }

    if API_VERBOSE_OUTPUT:
        packet["data_raw"] = raw_data

    return packet


def _build_fall_packet(fall_result: dict, source_topic: str = "sensor/fall_raw") -> dict:
    """
    Build fall_update packet — chỉ chứa kết quả fall detection.
    """
    return {
        "type": "fall_update",
        "source_topic": source_topic,
        "server_timestamp": int(time.time()),
        "fall": {
            "detected": fall_result.get("detected", False),
            "confidence": fall_result.get("confidence", 0.0),
            "num_samples": fall_result.get("num_samples"),
            "window_size": fall_result.get("window_size"),
            "pre_samples": fall_result.get("pre_samples"),
            "reason": fall_result.get("reason"),
            "status": fall_result.get("status", "model_result"),
        },
    }


def _get_fall_model():
    global _fall_model
    if _fall_model is None:
        with _fall_model_lock:
            if _fall_model is None:
                _fall_model = create_fall_model()
                model_name = getattr(_fall_model, "model_name", "unknown")
                print(f"🤖 Fall model da nap: {model_name}")
    return _fall_model


def _append_history(packet: dict):
    """
    Ghi health_update vào HISTORY_FILE (health_history.jsonl).
    Lưu đầy đủ: saved_at + toàn bộ fields (type, source_topic,
    server_timestamp, data.bpm, data.spo2, data.temp, data.status, data.ts).
    """
    append_jsonl_record(HISTORY_FILE, packet, lock=_history_lock)


def _append_fall_history(packet: dict):
    """
    Ghi fall_update vào FALL_HISTORY_FILE (fall_history.jsonl).
    Lưu đầy đủ: saved_at + toàn bộ fields (type, source_topic,
    server_timestamp, fall.detected, fall.confidence).
    """
    append_jsonl_record(FALL_HISTORY_FILE, packet, lock=_history_lock)


def _handle_alert_over_3_batches(health_samples):

    global _last_alert_time
    global _buzzer_active
    global _wait_next_normal_batch

    if not health_samples:
        return

    last_bpm = None
    last_spo2 = None
    last_temp = None

    for sample in health_samples:

        if _is_valid_vital_value(
            "heart_rate",
            getattr(sample, "heart_rate", None)
        ):
            last_bpm = float(sample.heart_rate)

        if _is_valid_vital_value(
            "spo2",
            getattr(sample, "spo2", None)
        ):
            last_spo2 = float(sample.spo2)

        if _is_valid_vital_value(
            "temp",
            getattr(sample, "temp", None)
        ):
            last_temp = float(sample.temp)

    current_batch = {
        "bpm": _is_vital_abnormal("bpm", last_bpm),
        "spo2": _is_vital_abnormal("spo2", last_spo2),
        "temp": _is_vital_abnormal("temp", last_temp),
    }

    with _batch_alert_lock:

        _vital_batch_buffer.append(current_batch)

        if len(_vital_batch_buffer) < 3:
            return

        b1 = _vital_batch_buffer[0]
        b2 = _vital_batch_buffer[1]
        b3 = _vital_batch_buffer[2]

    latest_normal = not any(b3.values())

    if latest_normal:
        if _buzzer_active:
            print("✅ Batch mới nhất NORMAL -> tắt còi")
            _stop_buzzer()
        return

    if _buzzer_active:
        return

    b1_abnormal = any(b1.values())
    b2_abnormal = any(b2.values())
    b3_abnormal = any(b3.values())

    if not (b1_abnormal and b2_abnormal and b3_abnormal):
        return

    triggered = [key for key, is_abnormal in b3.items() if is_abnormal]

    now = time.time()

    if now - _last_alert_time < ALERT_COOLDOWN_S:
        return

    _last_alert_time = now

    print(f"🚨 ALERT 3-BATCH: {triggered}")

    _buzzer_active = True
    _fire_beep_pattern()


def _read_history(limit: int = 50) -> list:
    """Đọc lịch sử health_update từ HISTORY_FILE, mới nhất trước."""
    return read_jsonl_records(HISTORY_FILE, limit=limit, lock=_history_lock)


def _read_fall_history(limit: int = 50) -> list:
    """Đọc lịch sử fall_update từ FALL_HISTORY_FILE, mới nhất trước."""
    return read_jsonl_records(FALL_HISTORY_FILE, limit=limit, lock=_history_lock)


def _make_training_id(prefix: str) -> str:
    return f"{prefix}-{int(time.time() * 1000)}"


def _normalize_training_label(label) -> str:
    normalized = str(label or "").strip().lower()
    if normalized not in VALID_TRAINING_LABELS:
        raise ValueError("label chi duoc la 'fall' hoac 'not_fall'")
    return normalized


def _set_pending_training_label(label: str) -> str:
    normalized = _normalize_training_label(label)
    global _pending_training_label
    with _pending_training_label_lock:
        _pending_training_label = normalized
    return normalized


def _consume_pending_training_label() -> str | None:
    global _pending_training_label
    with _pending_training_label_lock:
        label = _pending_training_label
        _pending_training_label = None
    return label


def _append_fall_raw_training_window(decoded_data: dict, result: dict | None = None) -> None:
    samples = decoded_data.get("data")
    if isinstance(samples, np.ndarray):
        samples = samples.astype(float).tolist()

    training_id = _make_training_id("fall")
    label = _consume_pending_training_label()
    record = {
        "training_id": training_id,
        "source_topic": "sensor/fall_raw",
        "received_at": int(time.time()),
        "trigger_ts": decoded_data.get("trigger_ts"),
        "num_samples": decoded_data.get("num_samples"),
        "pre_samples": decoded_data.get("pre_samples"),
        "reason": decoded_data.get("reason"),
        "data": samples,
        "model_result": result or {},
        "label": label,
    }
    append_jsonl_record(TRAINING_FALL_RAW_FILE, record, lock=_history_lock)
    return training_id


def _read_training_records(file_path: str) -> list[dict]:
    return read_jsonl_records(file_path, limit=1_000_000, lock=_history_lock)[::-1]


def _write_training_records(file_path: str, records: list[dict]) -> None:
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with _history_lock:
        with open(file_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _label_training_record(training_id: str, label: str) -> dict | None:
    normalized = _normalize_training_label(label)
    records = _read_training_records(TRAINING_FALL_RAW_FILE)
    updated = None

    for record in records:
        if record.get("training_id") == training_id:
            record["label"] = normalized
            record["label_updated_at"] = int(time.time())
            updated = record
            break

    if updated is None:
        return None

    _write_training_records(TRAINING_FALL_RAW_FILE, records)
    _save_fall_training_csv_file(records)
    return updated


def _label_latest_training_record(label: str) -> dict | None:
    records = _read_training_records(TRAINING_FALL_RAW_FILE)
    if not records:
        return None
    latest = records[-1]
    return _label_training_record(str(latest.get("training_id")), label)


def _csv_response(rows: list[dict], filename: str) -> Response:
    output = io.StringIO()
    if rows:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _flatten_fall_training_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        samples = record.get("data") or []
        result = record.get("model_result") or {}
        pre_samples = int(record.get("pre_samples") or 0)
        for idx, sample in enumerate(samples):
            values = list(sample) if isinstance(sample, (list, tuple)) else []
            values = (values + [None] * 8)[:8]
            rows.append({
                "training_id": record.get("training_id"),
                "saved_at": record.get("saved_at"),
                "trigger_ts": record.get("trigger_ts"),
                "sample_index": idx,
                "relative_index": idx - pre_samples,
                "is_pre_trigger": idx < pre_samples,
                "label": record.get("label"),
                "model_detected": result.get("detected"),
                "model_confidence": result.get("confidence"),
                "reason": record.get("reason"),
                "ax": values[0],
                "ay": values[1],
                "az": values[2],
                "gx": values[3],
                "gy": values[4],
                "gz": values[5],
                "acc_mag": values[6],
                "jerk": values[7],
            })
    return rows

def _save_fall_training_csv_file(records: list[dict]) -> None:
    rows = _flatten_fall_training_rows(records)

    if not rows:
        return

    import os
    os.makedirs(os.path.dirname(TRAINING_FALL_RAW_CSV_FILE), exist_ok=True)

    with open(TRAINING_FALL_RAW_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Da luu file CSV: {TRAINING_FALL_RAW_CSV_FILE}")

def _to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_non_negative_number(value):
    numeric = _to_number(value)
    if numeric is None or numeric < 0.0:
        return None
    return numeric


def _normalize_series_field(value, field_name, as_int=False):
    if isinstance(value, list):
        source = value
    elif value is None:
        return []
    else:
        source = [value]

    normalized = []
    for item in source:
        numeric = _to_number(item)
        if numeric is None:
            raise ValueError(f"Gia tri trong truong '{field_name}' khong hop le")
        normalized.append(int(round(numeric)) if as_int else float(numeric))

    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# Decoder Binary Fall_Raw từ ESP32 v6
# ══════════════════════════════════════════════════════════════════════════════
def _decode_fall_raw_binary(buf: bytes) -> dict | None:
    try:
        if len(buf) < 8:
            print(f"⚠️  Buffer quá nhỏ: {len(buf)} < 8 bytes")
            return None

        offset = 0
        ts = struct.unpack_from('>I', buf, offset)[0]
        offset += 4
        num_samples = struct.unpack_from('>H', buf, offset)[0]
        offset += 2
        pre_samples = buf[offset]
        offset += 1
        reason_len = buf[offset]
        offset += 1

        if offset + reason_len > len(buf):
            print(f"⚠️  Không đủ buffer cho reason string")
            return None

        reason = buf[offset:offset+reason_len].decode('utf-8', errors='replace')
        offset += reason_len

        samples = np.zeros((num_samples, 8), dtype=np.float32)

        SCALE_ACC = 1000.0
        SCALE_GYRO = 10.0
        SCALE_MAG = 1000.0
        SCALE_JERK = 1000.0

        for i in range(num_samples):
            if offset + 16 > len(buf):
                print(f"⚠️  Buffer không đủ tại sample {i}/{num_samples}")
                return None

            raw = struct.unpack_from('>8h', buf, offset)
            offset += 16

            samples[i, 0] = raw[0] / SCALE_ACC
            samples[i, 1] = raw[1] / SCALE_ACC
            samples[i, 2] = raw[2] / SCALE_ACC
            samples[i, 3] = raw[3] / SCALE_GYRO
            samples[i, 4] = raw[4] / SCALE_GYRO
            samples[i, 5] = raw[5] / SCALE_GYRO
            samples[i, 6] = raw[6] / SCALE_MAG
            samples[i, 7] = raw[7] / SCALE_JERK

        print(f"✅ Decode binary: {num_samples} samples × 8 channels | peak_idx={pre_samples} | reason='{reason}'")

        return {
            'trigger_ts': ts,
            'num_samples': num_samples,
            'pre_samples': pre_samples,
            'reason': reason,
            'data': samples
        }

    except Exception as e:
        print(f"❌ Lỗi decode binary fall_raw: {e}")
        traceback.print_exc()
        return None


def _process_fall_raw_with_model(decoded_data: dict, alert_data: dict | None = None) -> dict | None:

    try:
        model = _get_fall_model()
        samples_array = decoded_data['data']
        num_samples = decoded_data['num_samples']
        reason = decoded_data['reason']
        trigger_ts = decoded_data['trigger_ts']
        pre_samples = decoded_data['pre_samples']

        raw_window = np.asarray(samples_array, dtype=np.float32)
        window_size = int(getattr(model, "window_size", raw_window.shape[0]) or raw_window.shape[0])

        if raw_window.shape[0] == 0:
            print("fall_raw khong co sample")
            return None
        if raw_window.shape[0] < window_size:
            print(
                f"fall_raw ngan hon window model: {raw_window.shape[0]}/{window_size} samples, "
                "khong chay model de tranh du doan sai"
            )
            return {
                'detected': False,
                'confidence': 0.0,
                'num_samples': num_samples,
                'window_size': window_size,
                'pre_samples': pre_samples,
                'trigger_ts': trigger_ts,
                'reason': f"insufficient_samples:{reason}",
                'alert_trigger': "fall_raw_insufficient_samples",
                'status': "insufficient_samples",
            }

        def _select_window(raw_window_data: np.ndarray, center_idx, size: int) -> np.ndarray:
            if raw_window_data.shape[0] <= size:
                return raw_window_data

            try:
                center = int(center_idx)
            except (TypeError, ValueError):
                center = None

            if center is None or center < 0 or center >= raw_window_data.shape[0]:
                return raw_window_data[:size]

            start = max(0, center - size // 2)
            end = start + size
            if end > raw_window_data.shape[0]:
                end = raw_window_data.shape[0]
                start = end - size
            return raw_window_data[start:end]

        raw_window = _select_window(raw_window, pre_samples, window_size)
        print(f"✅ fall_raw du: {raw_window.shape[0]}/{window_size} samples → Chay model")

        with _latest_vitals_lock:
            hr = _latest_hr
            spo2 = _latest_spo2

        temp = _get_current_temperature()
        print(f"💚 Latest vitals từ sensor/data: HR={hr} BPM, SpO2={spo2:.1f}%, Temp={temp:.1f}°C")

        prediction = model.predict_raw_window(
            raw_window,
            vitals={"heart_rate": hr, "spo2": spo2, "temp": temp},
        )

        confidence = float(prediction.confidence)
        detected = bool(prediction.fall_detected)
        model_name = getattr(model, "model_name", "unknown")
        print(
            f"🧠 [MODEL:{model_name}] Inference: confidence={confidence:.3f} → detected={detected}"
        )

        alert_trigger = "fall_raw"
        if alert_data:
            alert_trigger = alert_data.get("trigger", alert_data.get("message", "unknown"))

        return {
            'detected': detected,
            'confidence': round(confidence, 3),
            'num_samples': num_samples,
            'window_size': window_size,
            'pre_samples': pre_samples,
            'trigger_ts': trigger_ts,
            'reason': reason,
            'alert_trigger': alert_trigger,
            'status': "model_result",
        }

    except Exception as e:
        print(f"❌ Lỗi xử lý fall_raw qua model: {e}")
        traceback.print_exc()
        return None


def _normalize_raw_payload_for_api(raw_payload):
    if not isinstance(raw_payload, dict):
        raise ValueError("Payload khong hop le, can doi tuong JSON")

    if isinstance(raw_payload.get("data"), list) and raw_payload.get("data"):
        series = raw_payload.get("data")
        times = []
        ir_vals = []
        red_vals = []
        for item in series:
            if not isinstance(item, dict):
                continue
            t = item.get("t") if "t" in item else item.get("ts", item.get("timestamp"))
            if t is not None:
                try:
                    times.append(int(t))
                except Exception:
                    times.append(None)
            else:
                times.append(None)

            ir_vals.append(int(item.get("ir")) if item.get("ir") is not None else 0)
            red_vals.append(int(item.get("red")) if item.get("red") is not None else 0)

        ts0_int = None
        ts_int = None
        valid_times = [t for t in times if isinstance(t, int)]
        if valid_times:
            ts0_int = valid_times[0]
            ts_int = valid_times[-1]

        fs = raw_payload.get("fs")
        sample_interval_ms = None
        try:
            if fs is not None:
                f = float(fs)
                if f > 0:
                    fs_interval_ms = max(1, int(round(1000.0 / f)))
                else:
                    fs_interval_ms = None
            else:
                fs_interval_ms = None
        except Exception:
            fs_interval_ms = None

        sample_interval_ms_time = None
        try:
            if len(valid_times) > 1:
                import numpy as _np
                diffs = _np.diff(_np.array(valid_times, dtype=np.int64))
                diffs = diffs[diffs > 0]
                if diffs.size > 0:
                    sample_interval_ms_time = int(max(1, int(_np.median(diffs))))
        except Exception:
            sample_interval_ms_time = None

        if sample_interval_ms_time is not None:
            sample_interval_ms = sample_interval_ms_time
            if fs_interval_ms is not None and fs_interval_ms > 0:
                diff_frac = abs(sample_interval_ms_time - fs_interval_ms) / float(fs_interval_ms)
                if diff_frac > 0.2:
                    print(f"⚠️  fs ({fs_interval_ms}ms) and timestamps-derived interval ({sample_interval_ms_time}ms) differ by {diff_frac*100:.0f}% - using timestamps")
        else:
            sample_interval_ms = fs_interval_ms

        return {
            "ts": ts_int if ts_int is not None else int(time.time() * 1000),
            "timestamp": ts_int if ts_int is not None else int(time.time() * 1000),
            "ts0": ts0_int,
            "temp": _to_non_negative_number(raw_payload.get("temp")),
            "sample_interval_ms": sample_interval_ms,
            "heart_rate": raw_payload.get("heart_rate", raw_payload.get("bpm")),
            "spo2": raw_payload.get("spo2"),
            "ir": ir_vals,
            "red": red_vals,
            "ax": None,
            "ay": None,
            "az": None,
            "gx": None,
            "gy": None,
            "gz": None,
        }

    ts = raw_payload.get("ts", raw_payload.get("timestamp"))
    ts_int = int(ts) if ts is not None else int(time.time() * 1000)
    ts0 = raw_payload.get("ts0")
    ts0_int = int(ts0) if ts0 is not None else None

    temp = _to_non_negative_number(raw_payload.get("temp"))
    ir_series = _normalize_series_field(raw_payload.get("ir"), "ir", as_int=True)
    red_series = _normalize_series_field(raw_payload.get("red"), "red", as_int=True)
    ax_series = _normalize_series_field(raw_payload.get("ax"), "ax")
    ay_series = _normalize_series_field(raw_payload.get("ay"), "ay")
    az_series = _normalize_series_field(raw_payload.get("az"), "az")
    gx_series = _normalize_series_field(raw_payload.get("gx"), "gx")
    gy_series = _normalize_series_field(raw_payload.get("gy"), "gy")
    gz_series = _normalize_series_field(raw_payload.get("gz"), "gz")

    explicit_interval = raw_payload.get("sample_interval_ms")
    inferred_interval = None
    if explicit_interval is not None:
        explicit_numeric = _to_non_negative_number(explicit_interval)
        if explicit_numeric is not None and explicit_numeric >= 1:
            inferred_interval = int(round(explicit_numeric))
    elif ts0_int is not None:
        series_len = max(
            len(ir_series), len(red_series),
            len(ax_series), len(ay_series), len(az_series),
            len(gx_series), len(gy_series), len(gz_series),
        )
        if series_len > 1 and ts_int > ts0_int:
            inferred_interval = max(1, int(round((ts_int - ts0_int) / float(series_len - 1))))

    return {
        "ts": ts_int,
        "timestamp": ts_int,
        "ts0": ts0_int,
        "temp": round(temp, 2) if temp is not None else None,
        "sample_interval_ms": inferred_interval,
        "heart_rate": raw_payload.get("heart_rate", raw_payload.get("bpm")),
        "spo2": raw_payload.get("spo2"),
        "quality": raw_payload.get("quality", {"valid": True, "status": "unknown"}),
        "ir": ir_series,
        "red": red_series,
        "ax": (None if not ax_series else (ax_series[0] if len(ax_series) == 1 else ax_series)),
        "ay": (None if not ay_series else (ay_series[0] if len(ay_series) == 1 else ay_series)),
        "az": (None if not az_series else (az_series[0] if len(az_series) == 1 else az_series)),
        "gx": (None if not gx_series else (gx_series[0] if len(gx_series) == 1 else gx_series)),
        "gy": (None if not gy_series else (gy_series[0] if len(gy_series) == 1 else gy_series)),
        "gz": (None if not gz_series else (gz_series[0] if len(gz_series) == 1 else gz_series)),
    }


def _normalize_health_data_for_frontend(raw_data):
    bpm = _to_non_negative_number(raw_data.get("bpm"))
    spo2 = _to_non_negative_number(raw_data.get("spo2"))
    temp = _to_non_negative_number(raw_data.get("temp"))
    ts = raw_data.get("ts", raw_data.get("timestamp"))

    return {
        "ts": int(ts) if ts is not None else None,
        "bpm": int(round(bpm)) if bpm is not None else None,
        "spo2": round(spo2, 1) if spo2 is not None else None,
        "temp": round(temp, 2) if temp is not None else None,
        "status": raw_data.get("status", "NORMAL"),
    }


def _get_current_temperature():
    global _current_temperature
    with _temp_lock:
        return _current_temperature


def _update_temperature(temp_value):
    global _current_temperature
    numeric = _to_non_negative_number(temp_value)
    if numeric is not None and 30.0 < numeric < 42.0:
        with _temp_lock:
            _current_temperature = numeric


def _build_stable_health_packet(raw_payload: dict, source_topic: str) -> dict:
    global _ema_bpm
    global _ema_spo2
    global _latest_hr
    global _latest_spo2

    quality = raw_payload.get("quality", {})
    quality_valid = _quality_is_valid(quality)
    quality_confidence = _quality_confidence(quality)
    signal_usable = _ppg_signal_is_usable(quality)
    quality_status = quality.get("status", "unknown") if isinstance(quality, dict) else "unknown"

    data = raw_payload.get("data", [])
    temp = raw_payload.get("temp")
    bpm_ppg = None
    spo2 = None
    fallback_bpm = None
    bpm_source = "missing"
    samples = []

    if not quality_valid:
        print(f"Signal quality bad: {quality_status} - using recent valid values only")
    elif not signal_usable:
        print(f"PPG signal weak despite valid quality: {quality_status} - using recent valid values only")
    elif not isinstance(data, list):
        print("PPG data is not a JSON array - using recent valid values only")
        data = []
    else:
        try:
            step_size = int(raw_payload.get("step_size", _PPG_STEP_SIZE) or _PPG_STEP_SIZE)
        except (TypeError, ValueError):
            step_size = _PPG_STEP_SIZE

        min_required_samples = min(max(80, step_size // 2), step_size)
        if len(data) < min_required_samples:
            print(f"Not enough PPG samples: {len(data)}/{min_required_samples}")
        else:
            for sample in data:
                if not isinstance(sample, dict):
                    continue
                t = sample.get("t") if "t" in sample else sample.get("ts")
                ir = sample.get("ir")
                red = sample.get("red")

                if t is None or ir is None or red is None:
                    continue

                try:
                    samples.append({
                        "t": int(t),
                        "ir": int(ir),
                        "red": int(red),
                    })
                except (TypeError, ValueError):
                    continue

            if len(samples) < min_required_samples:
                print(f"Not enough valid PPG samples: {len(samples)}/{min_required_samples}")
            else:
                fallback_bpm = calculate_bpm_from_samples(samples)

                try:
                    normalized_ppg = _normalize_raw_payload_for_api(raw_payload)
                    model_samples = from_json_samples(json.dumps(normalized_ppg))

                    if model_samples:
                        latest_sample = model_samples[-1]

                        if _is_valid_vital_value("heart_rate", getattr(latest_sample, "heart_rate", None)):
                            bpm_ppg = float(latest_sample.heart_rate)

                        if _is_valid_vital_value("spo2", getattr(latest_sample, "spo2", None)):
                            spo2 = float(latest_sample.spo2)

                        if _is_valid_vital_value("temp", getattr(latest_sample, "temp", None)):
                            temp = float(latest_sample.temp)

                except Exception as exc:
                    print(f"SpO2/BPM calculation error: {exc}")

                bpm_ppg, bpm_source = _choose_bpm_estimate(bpm_ppg, fallback_bpm)

    bpm_raw = bpm_ppg
    spo2_raw = spo2
    now_mono = time.monotonic()

    with _ema_lock:
        bpm_ppg, _ema_bpm, bpm_stale, bpm_updated, bpm_stable_input = _smooth_vital_value(
            key="heart_rate",
            raw_value=bpm_ppg,
            ema_value=_ema_bpm,
            recent_values=_recent_bpm_values,
            base_alpha=EMA_ALPHA_BPM,
            confidence=quality_confidence,
            jump_threshold=18.0,
            max_delta_per_update=3.0,
            decimals=1,
            now_mono=now_mono,
        )
        if bpm_updated:
            _last_valid_vitals["heart_rate"] = bpm_ppg
            _last_valid_vital_times["heart_rate"] = now_mono

        spo2, _ema_spo2, spo2_stale, spo2_updated, spo2_stable_input = _smooth_vital_value(
            key="spo2",
            raw_value=spo2,
            ema_value=_ema_spo2,
            recent_values=_recent_spo2_values,
            base_alpha=EMA_ALPHA_SPO2,
            confidence=quality_confidence,
            jump_threshold=2.5,
            max_delta_per_update=0.6,
            decimals=1,
            now_mono=now_mono,
        )
        if spo2_updated:
            _last_valid_vitals["spo2"] = spo2
            _last_valid_vital_times["spo2"] = now_mono

        if _is_valid_vital_value("temp", temp):
            temp_raw = round(float(temp), 2)
            _recent_temp_values.append(temp_raw)
            temp_avg = _last_average(_recent_temp_values)
            temp = round(temp_avg if temp_avg is not None else temp_raw, 2)
            temp_stale = False
            _last_valid_vitals["temp"] = temp
            _last_valid_vital_times["temp"] = now_mono
        else:
            temp, temp_stale = _get_recent_valid_value("temp", now_mono)

        ema_bpm = round(_ema_bpm, 1) if _ema_bpm is not None else None
        ema_spo2 = round(_ema_spo2, 1) if _ema_spo2 is not None else None

    _update_temperature(temp)

    with _latest_vitals_lock:
        if bpm_ppg is not None:
            _latest_hr = bpm_ppg
        if spo2 is not None:
            _latest_spo2 = spo2

    last_sample_ts = None
    if isinstance(data, list) and data and isinstance(data[-1], dict):
        last_sample_ts = data[-1].get("t", data[-1].get("ts"))
    ts = raw_payload.get("ts") or raw_payload.get("timestamp") or last_sample_ts
    status = classify(bpm_ppg, temp, spo2)

    print(
        f"Vitals BPM={bpm_ppg} raw={bpm_raw} stable={bpm_stable_input} "
        f"SpO2={spo2} raw={spo2_raw} Temp={temp} "
        f"EMA_BPM={ema_bpm} EMA_SpO2={ema_spo2} conf={quality_confidence:.2f} src={bpm_source}"
    )

    return {
        "type": "health_update",
        "source_topic": source_topic,
        "server_timestamp": int(time.time()),
        "data": {
            "ts": ts,
            "bpm": bpm_ppg,
            "spo2": spo2,
            "temp": temp,
            "status": status,
            "bpm_raw": bpm_raw,
            "spo2_raw": spo2_raw,
            "bpm_fallback_raw": fallback_bpm,
            "bpm_source": bpm_source,
            "bpm_stable_input": bpm_stable_input,
            "spo2_stable_input": spo2_stable_input,
            "confidence": quality_confidence,
            "signal_usable": signal_usable,
            "stale": {
                "bpm": bpm_stale,
                "spo2": spo2_stale,
                "temp": temp_stale,
            },
            "quality": raw_payload.get("quality", {}),
        },
    }


# ====== CALLBACK ======
def on_connect(client, userdata, flags, reason_code, properties=None):
    global _mqtt_connected
    if reason_code == 0:
        if not _mqtt_connected:
            print("✅ Da ket noi HiveMQ Cloud")
        _mqtt_connected = True

        client.subscribe("sensor/data")
        client.subscribe("sensor/fall_raw")
        client.subscribe("health/#")

        if not _mqtt_connected:
            print("📡 Da dang ky topic: sensor/#, health/#")
    else:
        print("❌ Ket noi that bai, ma loi:", reason_code)


def on_disconnect(client, userdata, rc, properties=None):
    global _mqtt_connected
    _mqtt_connected = False
    print(f"⚠️ MQTT disconnected, rc={rc}")


def on_message(client, userdata, msg):
    print("\n📩 ===== TIN NHAN MOI =====")
    print("📌 Chu de:", msg.topic)

    try:
        global _latest_health_packet
        global _latest_fall_packet
        global _latest_raw_payload
        global _waiting_for_fall_raw
        global _latest_hr
        global _latest_spo2

        # ════════════════════════════════════════════════════════════════════════
        # [LUỒNG 2] SENSOR FALL_RAW → chỉ emit "fall_update"
        # ════════════════════════════════════════════════════════════════════════
        if msg.topic == "sensor/fall_raw":
            with _fall_raw_state_lock:
                _waiting_for_fall_raw = True
                pending_alert = None

            try:
                decoded = _decode_fall_raw_binary(msg.payload)
                if not decoded:
                    print("❌ Không thể decode fall_raw binary")
                    return

                if FALL_MODEL_BLOCK_MS > 0:
                    print(f"⏸️  Tạm dừng {FALL_MODEL_BLOCK_MS}ms để xử lý fall model")
                    time.sleep(FALL_MODEL_BLOCK_MS / 1000.0)

                print(f"🧠 [MODEL] Đang xử lý {decoded['num_samples']} motion samples...")
                result = _process_fall_raw_with_model(decoded, pending_alert)
                training_id = _append_fall_raw_training_window(decoded, result)
                print(f"[TRAINING] Saved window: {training_id}")

                records = _read_training_records(TRAINING_FALL_RAW_FILE)
                _save_fall_training_csv_file(records)

                if result:
                    if result['detected']:
                        print(f"🚨 [RESULT] TẾ NGÃ XÁC NHẬN! | Confidence: {result['confidence']:.3f}")
                    else:
                        print(f"✓ [RESULT] Không phải ngã | Confidence: {result['confidence']:.3f}")

                    # Build và emit RIÊNG packet fall_update
                    fall_packet = _build_fall_packet(result, source_topic="sensor/fall_raw")

                    with _packet_lock:
                        _latest_fall_packet = fall_packet

                    _append_fall_history(fall_packet)
                    _emit_fall(fall_packet, delay_ms=WS_FALL_EMIT_DELAY_MS)

                    print(f"📡 [FALL_UPDATE] detected={result['detected']} confidence={result['confidence']:.3f}")

                    # Kích hoạt buzzer nếu ngã được xác nhận
                    if result['detected']:
                        _send_device_command("buzzer_on", duration_ms=5000)
                else:
                    print("⚠️  Model không trả về kết quả")

            finally:
                with _fall_raw_state_lock:
                    _waiting_for_fall_raw = False

            return

        # ════════════════════════════════════════════════════════════════════════
        # [LUỒNG 1] SENSOR DATA → chỉ emit "health_update"
        # ════════════════════════════════════════════════════════════════════════

        if SENSOR_INPUT_PROCESS_DELAY_MS > 0:
            time.sleep(SENSOR_INPUT_PROCESS_DELAY_MS / 1000.0)

        try:
            raw = msg.payload.decode(errors='replace')
        except Exception as e:
            print(f"❌ Payload khong hop le: loi giai ma binary - {e}")
            return

        try:
            raw_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Payload JSON khong hop le") from exc

        packet = _build_stable_health_packet(raw_payload, msg.topic)
        with _packet_lock:
            _latest_health_packet = packet
        _append_history(packet)
        _emit_health(packet, delay_ms=WS_HEALTH_EMIT_DELAY_MS)
        print(
            f"HEALTH_UPDATE BPM={packet['data']['bpm']} "
            f"SpO2={packet['data']['spo2']} Temp={packet['data']['temp']}"
        )
        return

    except ValueError as exc:
        print("❌ Payload khong hop le:", exc)
        socketio.emit(
            "health_error",
            {
                "type": "health_error",
                "source_topic": msg.topic,
                "message": str(exc),
            },
        )
    except UnicodeDecodeError:
        print("❌ Khong giai ma duoc tin nhan")


def _send_device_command(cmd: str, **kwargs):
    global _mqtt_client
    if _mqtt_client is None:
        print("⚠️ MQTT client not connected, cannot send command")
        return

    payload = {"cmd": cmd, **kwargs}
    try:
        with _mqtt_client_lock:
            result = _mqtt_client.publish(
                "device/command",
                json.dumps(payload),
                qos=1
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📤 Command sent: {cmd}")
            else:
                print(f"❌ Failed to send command: {result.rc}")
    except Exception as exc:
        print(f"❌ Error sending command: {exc}")


def _is_vital_abnormal(key: str, value) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    thresholds = VITAL_THRESHOLDS.get(key, {})
    low  = thresholds.get("low")
    high = thresholds.get("high")
    if low is not None and v < low:
        return True
    if high is not None and v > high:
        return True
    return False


def _cancel_alert_timers():
    with _alert_timers_lock:
        for t in _alert_timers:
            try:
                t.cancel()
            except Exception:
                pass
        _alert_timers.clear()


def _fire_beep_pattern():
    cycle_ms = ALERT_BEEP_ON_MS + ALERT_BEEP_OFF_MS
    total_ms = ALERT_BEEP_DURATION_S * 1000
    count    = max(1, int(total_ms / cycle_ms))

    _cancel_alert_timers()

    with _alert_timers_lock:
        for i in range(count):
            delay_s = i * cycle_ms / 1000.0
            t = Timer(
                delay_s,
                _send_device_command,
                kwargs={"cmd": "buzzer_on", "duration_ms": ALERT_BEEP_ON_MS},
            )
            t.daemon = True
            _alert_timers.append(t)
            t.start()

    print(
        f"🔔 [BUZZER] Pattern started: {count} beeps × {ALERT_BEEP_ON_MS}ms ON / "
        f"{ALERT_BEEP_OFF_MS}ms OFF, total ~{ALERT_BEEP_DURATION_S}s"
    )


def _stop_buzzer():
    global _buzzer_active
    _cancel_alert_timers()
    _send_device_command("buzzer_off")
    _buzzer_active = False
    print("🔕 [BUZZER] Stopped")


def build_mqtt_client():
    if mqtt is None:
        raise RuntimeError("Thieu paho-mqtt. Cai dat bang: pip install -r requirements.txt")

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=CLIENT_ID,
    )
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(ca_certs=certifi.where(), tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.on_connect = on_connect
    client.on_message = on_message
    return client


# ════════════════════════════════════════════════════════════════════════════════
# WEBSOCKET SOCKET.IO HANDLERS
# ════════════════════════════════════════════════════════════════════════════════

@socketio.on("buzz")
def handle_buzz(data):
    """
    Socket.IO event handler để tắt coi (buzzer).
    Frontend gửi event "buzz" để tắt báo động.
    
    Dữ liệu nhận được:
        {
            "action": "off",  # hoặc "on"
            "timestamp": unix_seconds,
            "reason": "user_dismissed" hoặc lý do khác
        }
    
    Xử lý:
    1. Tắt buzzer qua device/command
    2. Gửi alert message tới sensor/alert
    3. Phát lại confirm về frontend
    """
    try:
        print(f"📡 Socket.IO 'buzz' event received: {data}")
        
        # Lấy dữ liệu từ client
        action = data.get("action", "off") if isinstance(data, dict) else "off"
        reason = data.get("reason", "unknown") if isinstance(data, dict) else "unknown"
        client_timestamp = data.get("timestamp") if isinstance(data, dict) else None
        
        # Tắt buzzer
        if action == "off":
            global _buzzer_active
            _stop_buzzer()
            print(f"🔕 Buzzer stopped. Reason: {reason}")
        
        # Gửi dữ liệu đến sensor/alert topic
        alert_payload = {
            "type": "buzz",
            "action": action,
            "reason": reason,
            "server_timestamp": int(time.time()),
            "client_timestamp": client_timestamp,
        }
        
        _send_device_command("buzz", action=action, reason=reason)
        
        # Publish tới MQTT sensor/alert
        global _mqtt_client
        if _mqtt_client is not None:
            try:
                with _mqtt_client_lock:
                    result = _mqtt_client.publish(
                        "sensor/alert",
                        json.dumps(alert_payload),
                        qos=1
                    )
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print(f"📤 Alert sent to sensor/alert: {alert_payload}")
                    else:
                        print(f"❌ Failed to publish alert: {result.rc}")
            except Exception as exc:
                print(f"❌ Error publishing to sensor/alert: {exc}")
        
        # Gửi xác nhận về frontend (broadcast)
        socketio.emit(
            "buzz_response",
            {
                "type": "buzz_response",
                "status": "success",
                "action": action,
                "server_timestamp": int(time.time()),
            },
        )
        print("✅ Buzz response sent to frontend")
        
    except Exception as exc:
        print(f"❌ Error handling buzz event: {exc}")
        traceback.print_exc()
        socketio.emit(
            "buzz_response",
            {
                "type": "buzz_response",
                "status": "error",
                "message": str(exc),
            },
        )


@app.get("/api/docs")
def get_api_docs():
    return jsonify(
        {
            "name": "Heart Rate Monitor Backend API",
            "version": "2.0",
            "base_url_hint": "/api",
            "http_endpoints": [
                {
                    "method": "GET",
                    "path": "/api/docs",
                    "description": "Tai lieu API cho frontend.",
                },
                {
                    "method": "GET",
                    "path": "/api/history",
                    "description": "Lich su ket hop health_update va fall_update. ?limit=1..1000 | ?type=all|health|fall",
                    "response": "{ count, items: [{ saved_at, type, source_topic, server_timestamp, ...data_or_fall }] }",
                },
                {
                    "method": "GET",
                    "path": "/api/training",
                    "description": "Thong ke du lieu fall_raw da thu de train tiep.",
                },
                {
                    "method": "GET",
                    "path": "/api/training/fall.csv",
                    "description": "Tai CSV du lieu te nga da qua filter .ino.",
                },
            ],
            "websocket": {
                "transport": "Socket.IO",
                "events_received": [
                    {
                        "event": "buzz",
                        "description": "Tắt coi/báo động từ frontend.",
                        "payload_shape": {
                            "action": "off|on",
                            "reason": "user_dismissed|timeout|emergency_stop|etc",
                            "timestamp": "unix_seconds_or_null",
                        },
                        "handling": "Tắt buzzer và gửi alert tới sensor/alert",
                    },
                ],
                "events_emitted": [
                    {
                        "event": "health_update",
                        "description": "Dữ liệu sinh hiệu (BPM/SpO2/Temp/Status) từ sensor/data. KHÔNG chứa fall.",
                        "source": "sensor/data",
                        "payload_shape": {
                            "type": "health_update",
                            "source_topic": "sensor/data",
                            "server_timestamp": "unix_seconds",
                            "data": {
                                "ts": "number",
                                "bpm": "number",
                                "spo2": "number_or_null",
                                "temp": "number",
                                "status": "NORMAL|FEVER|LOW_TEMP|HIGH_HEART_RATE|LOW_HEART_RATE|LOW_SPO2",
                            },
                        },
                    },
                    {
                        "event": "fall_update",
                        "description": "Kết quả phát hiện ngã từ sensor/fall_raw. KHÔNG chứa vitals.",
                        "source": "sensor/fall_raw",
                        "payload_shape": {
                            "type": "fall_update",
                            "source_topic": "sensor/fall_raw",
                            "server_timestamp": "unix_seconds",
                            "fall": {
                                "detected": "bool",
                                "confidence": "float_0_to_1",
                            },
                        },
                    },
                    {
                        "event": "buzz_response",
                        "description": "Phản hồi từ server sau khi nhận event 'buzz' từ frontend.",
                        "payload_shape": {
                            "type": "buzz_response",
                            "status": "success|error",
                            "action": "off|on",
                            "server_timestamp": "unix_seconds",
                            "message": "error_message_or_null",
                        },
                    },
                    {
                        "event": "health_error",
                        "description": "Loi parse/validate payload MQTT.",
                    },
                ],
            },
            "notes": [
                "health_update và fall_update là 2 luồng độc lập — subscribe cả hai để có đầy đủ thông tin.",
                "fall_update chỉ được emit khi sensor/fall_raw gửi dữ liệu (không gửi liên tục).",
                "Gia tri BPM co the la 0 khi tin hieu IR khong du chat luong de detect peak.",
                "buzz event: frontend gửi để tắt coi, server gửi lại buzz_response.",
            ],
        }
    )


def _parse_limit(default: int = 50):
    """Parse ?limit= query param. Returns (limit, error_response|None)."""
    raw = request.args.get("limit", str(default))
    try:
        val = int(raw)
    except ValueError:
        return None, (jsonify({"message": "Tham so 'limit' phai la so nguyen"}), 400)
    if val < 1 or val > 1000:
        return None, (jsonify({"message": "Tham so 'limit' phai trong khoang 1..1000"}), 400)
    return val, None


@app.get("/api/history")
def get_history():
    """
    Lịch sử kết hợp health_update và fall_update, mới nhất trước.
    Query params:
      ?limit=50          số bản ghi trả về (mặc định 50, tối đa 1000)
      ?type=all          health + fall (mặc định)
      ?type=health       chỉ health_update
      ?type=fall         chỉ fall_update
    """
    limit, err = _parse_limit()
    if err:
        return err

    record_type = request.args.get("type", "all")

    if record_type == "health":
        items = _read_history(limit=limit)
    elif record_type == "fall":
        items = _read_fall_history(limit=limit)
    else:
        health_items = _read_history(limit=limit)
        fall_items   = _read_fall_history(limit=limit)
        merged = health_items + fall_items
        merged.sort(key=lambda r: r.get("saved_at", 0), reverse=True)
        items = merged[:limit]

    return jsonify({"count": len(items), "items": items})


@app.get("/api/training")
def get_training_summary():
    if not FALL_TRAINING_EXPORT_ENABLED:
        return jsonify({"message": "Fall training export da bi khoa"}), 403

    fall_records = _read_training_records(TRAINING_FALL_RAW_FILE)
    return jsonify(
        {
            "fall_raw_windows": len(fall_records),
            "csv": {
                "fall": "/api/training/fall.csv",
            },
            "storage": {
                "fall_jsonl": TRAINING_FALL_RAW_FILE,
            },
        }
    )


@app.get("/api/training/fall.csv")
def export_fall_training_csv():
    if not FALL_TRAINING_EXPORT_ENABLED:
        return jsonify({"message": "Fall training export da bi khoa"}), 403

    records = _read_training_records(TRAINING_FALL_RAW_FILE)
    return _csv_response(_flatten_fall_training_rows(records), "fall_raw_training_windows.csv")


def main():
    global _mqtt_client

    _mqtt_client = build_mqtt_client()
    mqtt_started = False

    try:
        try:
            _mqtt_client.connect_async(BROKER, MQTT_PORT)
            _mqtt_client.loop_start()
            mqtt_started = True
            print("✅ MQTT loop da khoi dong (async)")
        except Exception as exc:
            print("⚠️ Khong the ket noi MQTT luc khoi dong:", exc)
            if MQTT_REQUIRED:
                raise

        print(f"🚀 API da khoi dong (bind): http://{API_BIND_HOST}:{API_PORT}")
        print(f"🌐 Truy cap tren may nay: http://{API_ACCESS_HOST}:{API_PORT}")
        print("🔌 WebSocket events: health_update | fall_update")
        socketio.run(
            app,
            host=API_BIND_HOST,
            port=API_PORT,
            allow_unsafe_werkzeug=True,
        )
    except Exception:
        print("❌ Loi khoi dong backend:")
        traceback.print_exc()
        raise
    finally:
        if mqtt_started and _mqtt_client:
            _mqtt_client.loop_stop()
            _mqtt_client.disconnect()




if __name__ == "__main__":
    main()
