import paho.mqtt.client as mqtt
import ssl
import certifi
import time
import os
import json
import traceback
import struct
from threading import Lock, Timer
from collections import deque

import numpy as np

from flask import Flask, jsonify, request
from flask_socketio import SocketIO

from fall_model import create_fall_model
from model import from_json_samples, classify, STATUS_FALL_DETECTED

_buzzer_active = False
_wait_next_normal_batch = False
# Lưu lịch sử nhiều batch
# Buffer lưu 3 batch gần nhất
_vital_batch_buffer = deque(maxlen=3)

# Cooldown tránh spam
_last_alert_time = 0
ALERT_COOLDOWN_S = 20

_batch_alert_lock = Lock()


VITAL_THRESHOLDS = {
    "bpm": {
        "low":  int(os.getenv("ALERT_BPM_LOW",  "50")),   # < 50 → bradycardia
        "high": int(os.getenv("ALERT_BPM_HIGH", "120")),  # > 120 → tachycardia
    },
    "spo2": {
        "low":  float(os.getenv("ALERT_SPO2_LOW",  "93.0")),  # < 93 → nguy hiểm
        "high": float(os.getenv("ALERT_SPO2_HIGH", "100.1")), # SpO2 không có high thực tế
    },
    "temp": {
        "low":  float(os.getenv("ALERT_TEMP_LOW",  "32.5")),  # < 32.5 → hạ thân nhiệt
        "high": float(os.getenv("ALERT_TEMP_HIGH", "39.5")),  # > 39.5 → sốt
    },
}
 
# Tần số beep: ON ms, OFF ms, số lần lặp trong 60 giây
ALERT_BEEP_ON_MS    = max(100, int(os.getenv("ALERT_BEEP_ON_MS",   "500")))
ALERT_BEEP_OFF_MS   = max(100, int(os.getenv("ALERT_BEEP_OFF_MS",  "700")))
ALERT_BEEP_DURATION_S = max(10, int(os.getenv("ALERT_BEEP_DURATION_S", "60")))
 
# State machine: theo dõi trạng thái trước của từng vital
# Giá trị: True = đang abnormal, False = đang normal
_vital_prev_abnormal: dict[str, bool] = {
    "bpm":  False,
    "spo2": False,
    "temp": False,
}
_alert_state_lock = Lock()
 
# Danh sách timer đang chạy (để cancel khi có alert mới)
_alert_timers: list[Timer] = []
_alert_timers_lock = Lock()

BROKER = "11060dbd13b54fc988ae8f9bfc43c089.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
USERNAME = "heart-rate"
PASSWORD = "aB123456"
CLIENT_ID = "python_backend1dsdssdaassdasdd"
MQTT_REQUIRED = os.getenv("MQTT_REQUIRED", "false").lower() == "true"
API_BIND_HOST = os.getenv("API_BIND_HOST", "0.0.0.0")
API_ACCESS_HOST = os.getenv("API_ACCESS_HOST", "192.168.1.23")
HISTORY_FILE        = os.getenv("HISTORY_FILE",      "health_history.jsonl")
FALL_HISTORY_FILE   = os.getenv("FALL_HISTORY_FILE", "fall_history.jsonl")
USE_AI_FALL_MODEL = os.getenv("USE_AI_FALL_MODEL", "true").lower() == "true"
API_VERBOSE_OUTPUT = os.getenv("API_VERBOSE_OUTPUT", "false").lower() == "true"


def _resolve_api_port() -> int:
    raw_port = os.getenv("API_PORT") or os.getenv("PORT") or "5050"
    try:
        return int(raw_port)
    except ValueError:
        return 5050


API_PORT = _resolve_api_port()
SENSOR_INPUT_PROCESS_DELAY_MS = max(0, int(os.getenv("SENSOR_INPUT_PROCESS_DELAY_MS", "0")))
WS_HEALTH_EMIT_DELAY_MS = max(0, int(os.getenv("WS_HEALTH_EMIT_DELAY_MS", "120")))
WS_FALL_EMIT_DELAY_MS = max(0, int(os.getenv("WS_FALL_EMIT_DELAY_MS", "0")))
FALL_MODEL_BLOCK_MS = max(0, int(os.getenv("FALL_MODEL_BLOCK_MS", "1000")))
BUZZER_BEEP_ON_MS = max(0, int(os.getenv("BUZZER_BEEP_ON_MS", "2000")))
BUZZER_BEEP_OFF_MS = max(0, int(os.getenv("BUZZER_BEEP_OFF_MS", "1000")))
BUZZER_BEEP_COUNT = max(1, int(os.getenv("BUZZER_BEEP_COUNT", "2")))
BUZZER_TOTAL_MS = max(0, int(os.getenv("BUZZER_TOTAL_MS", "60000")))

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

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


def _is_valid_vital_value(key: str, value):
    try:
        if value is None:
            return False
        if key == "heart_rate":
            v = float(value)
            return v > 0 and v < 300
        if key == "spo2":
            v = float(value)
            return 50.0 <= v <= 200.0
        if key == "temp":
            v = float(value)
            return 25.0 <= v <= 45.0
    except Exception:
        return False
    return False


# ── Trạng thái luồng Fall_Raw ────────────────────────────────────────────────
_waiting_for_fall_raw = False
_fall_raw_state_lock = Lock()


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
        },
    }


def _get_fall_model():
    global _fall_model
    if _fall_model is None:
        with _fall_model_lock:
            if _fall_model is None:
                _fall_model = create_fall_model()
                print("🤖 Fall model da nap: AI(model.h5)")
    return _fall_model


def _append_history(packet: dict):
    """
    Ghi health_update vào HISTORY_FILE (health_history.jsonl).
    Lưu đầy đủ: saved_at + toàn bộ fields (type, source_topic,
    server_timestamp, data.bpm, data.spo2, data.temp, data.status, data.ts).
    """
    history_line = {"saved_at": int(time.time()), **packet}
    with _history_lock:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(history_line, ensure_ascii=False) + "\n")


def _append_fall_history(packet: dict):
    """
    Ghi fall_update vào FALL_HISTORY_FILE (fall_history.jsonl).
    Lưu đầy đủ: saved_at + toàn bộ fields (type, source_topic,
    server_timestamp, fall.detected, fall.confidence).
    """
    history_line = {"saved_at": int(time.time()), **packet}
    with _history_lock:
        with open(FALL_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(history_line, ensure_ascii=False) + "\n")


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
    if limit <= 0 or not os.path.exists(HISTORY_FILE):
        return []
    with _history_lock:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-limit:][::-1]


def _read_fall_history(limit: int = 50) -> list:
    """Đọc lịch sử fall_update từ FALL_HISTORY_FILE, mới nhất trước."""
    if limit <= 0 or not os.path.exists(FALL_HISTORY_FILE):
        return []
    with _history_lock:
        with open(FALL_HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-limit:][::-1]


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

        if samples_array.shape[0] < model._required_window:
            print(f"⏳ fall_raw chua du: {samples_array.shape[0]}/{model._required_window} samples")
            return None
        if samples_array.shape[0] > model._required_window:
            samples_array = samples_array[:model._required_window]

        motion_buf = samples_array
        print(f"✅ fall_raw du: {motion_buf.shape[0]}/{model._required_window} samples → Chay model")

        with _latest_vitals_lock:
            hr = _latest_hr
            spo2 = _latest_spo2

        temp = _get_current_temperature()
        print(f"💚 Latest vitals từ sensor/data: HR={hr} BPM, SpO2={spo2:.1f}%, Temp={temp:.1f}°C")

        features_array = np.zeros((motion_buf.shape[0], 11), dtype=np.float32)

        for i in range(motion_buf.shape[0]):
            ax, ay, az = motion_buf[i, 0], motion_buf[i, 1], motion_buf[i, 2]
            gx, gy, gz = motion_buf[i, 3], motion_buf[i, 4], motion_buf[i, 5]

            roll = np.arctan2(ay, az + 1e-6)
            pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + 1e-6)

            features_array[i] = [ax, ay, az, gx, gy, gz, roll, pitch, hr, spo2, temp]

        mean = model._mean
        std = model._std
        normalized = (features_array - mean) / (std + 1e-6)

        X = normalized[np.newaxis, ...]

        pred = model._model.predict(X, verbose=0)
        confidence = float(pred[0][0])

        threshold = float(os.getenv("AI_FALL_THRESHOLD", "0.6"))
        detected = confidence >= threshold

        print(f"🧠 [MODEL] Features=11 (150 motion) | Inference: confidence={confidence:.3f}, threshold={threshold} → detected={detected}")

        alert_trigger = "fall_raw"
        if alert_data:
            alert_trigger = alert_data.get("trigger", alert_data.get("message", "unknown"))

        return {
            'detected': detected,
            'confidence': round(confidence, 3),
            'num_samples': num_samples,
            'pre_samples': pre_samples,
            'trigger_ts': trigger_ts,
            'reason': reason,
            'alert_trigger': alert_trigger,
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

        normalized_raw_payload = _normalize_raw_payload_for_api(raw_payload)

        _update_temperature(raw_payload.get("temp"))

        health_samples = from_json_samples(json.dumps(normalized_raw_payload))

        latest_health_packet = None
        for health_data in health_samples:
            with _latest_vitals_lock:
                hr_for_status = (
                    health_data.heart_rate
                    if _is_valid_vital_value("heart_rate", health_data.heart_rate)
                    else _last_valid_vitals.get("heart_rate")
                )
                temp_for_status = (
                    health_data.temp
                    if _is_valid_vital_value("temp", health_data.temp)
                    else _last_valid_vitals.get("temp")
                )

            status = classify(hr_for_status, temp_for_status, health_data.spo2)

            # Build health packet (vitals only, không có fall)
            packet = _build_health_packet(health_data, msg.topic)
            packet["data"]["status"] = status
            latest_health_packet = packet

            with _latest_vitals_lock:
                if _is_valid_vital_value("heart_rate", health_data.heart_rate):
                    _last_valid_vitals["heart_rate"] = float(health_data.heart_rate)
                if _is_valid_vital_value("spo2", health_data.spo2):
                    _last_valid_vitals["spo2"] = float(health_data.spo2)
                if _is_valid_vital_value("temp", health_data.temp):
                    _last_valid_vitals["temp"] = float(health_data.temp)

        _handle_alert_over_3_batches(health_samples)

        if latest_health_packet is not None:
            with _packet_lock:
                # Giữ lại BPM hợp lệ nếu batch mới báo 0/None
                try:
                    bpm_val = latest_health_packet.get("data", {}).get("bpm")
                except Exception:
                    bpm_val = None

                if bpm_val is None or bpm_val == 0:
                    if _latest_hr is not None and _latest_hr > 0:
                        latest_health_packet["data"]["bpm"] = _latest_hr
                        bpm_val = _latest_hr

                if bpm_val is not None:
                    try:
                        _latest_hr = float(bpm_val)
                    except Exception:
                        pass

                try:
                    spo2_val = latest_health_packet.get("data", {}).get("spo2")
                except Exception:
                    spo2_val = None

                if spo2_val is None:
                    if _latest_spo2 is not None and _latest_spo2 > 0:
                        latest_health_packet["data"]["spo2"] = _latest_spo2
                        spo2_val = _latest_spo2

                if spo2_val is not None:
                    try:
                        _latest_spo2 = float(spo2_val)
                    except Exception:
                        pass

                _latest_health_packet = latest_health_packet
                _latest_raw_payload = normalized_raw_payload

            with _fall_raw_state_lock:
                waiting_for_fall_raw = _waiting_for_fall_raw

            if waiting_for_fall_raw:
                print("⏳ [LUỒNG 1] Dang cho fall_raw, tam ngung emit health_update")
            else:
                _append_history(latest_health_packet)
                _emit_health(latest_health_packet, delay_ms=WS_HEALTH_EMIT_DELAY_MS)

                print(
                    f"💓 [HEALTH_UPDATE] BPM={latest_health_packet['data']['bpm']} "
                    f"SpO2={latest_health_packet['data']['spo2']}% "
                    f"Temp={latest_health_packet['data']['temp']}°C"
                )

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