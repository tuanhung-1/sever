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
_window_lock = Lock()


VITAL_THRESHOLDS = {
    "bpm": {
        "low":  int(os.getenv("ALERT_BPM_LOW",  "50")),   # < 50 → bradycardia
        "high": int(os.getenv("ALERT_BPM_HIGH", "120")),  # > 120 → tachycardia
    },
    "spo2": {
        "low":  float(os.getenv("ALERT_SPO2_LOW",  "90.0")),  # < 90 → nguy hiểm
        "high": float(os.getenv("ALERT_SPO2_HIGH", "100.1")), # SpO2 không có high thực tế
    },
    "temp": {
        "low":  float(os.getenv("ALERT_TEMP_LOW",  "35.0")),  # < 35 → hạ thân nhiệt
        "high": float(os.getenv("ALERT_TEMP_HIGH", "37.5")),  # > 37.5 → sốt
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
CLIENT_ID = "python_backend1dsdssdaasdasdd"
MQTT_REQUIRED = os.getenv("MQTT_REQUIRED", "false").lower() == "true"
API_BIND_HOST = os.getenv("API_BIND_HOST", "0.0.0.0")
API_ACCESS_HOST = os.getenv("API_ACCESS_HOST", "192.168.1.23")
HISTORY_FILE = os.getenv("HISTORY_FILE", "health_history.jsonl")
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

_latest_packet = None
_latest_raw_payload = None
_packet_lock = Lock()
_history_lock = Lock()
_fall_model_lock = Lock()
_fall_model = None
_mqtt_client = None
_mqtt_client_lock = Lock()
_mqtt_connected = False
_current_temperature = 36.5  # Nhiệt độ cơ thể mặc định (lấy từ sensor/data)
_temp_lock = Lock()
_emit_timer = None  # Throttle timer cho Socket.IO emit
_emit_timer_lock = Lock()



# ── Lưu latest vitals từ sensor/data (chỉ 1 value) ────────────────────────────
_latest_hr = 0.0          # Latest heart rate (BPM) từ sensor/data
_latest_spo2 = 0.0        # Latest SpO2 (%) từ sensor/data
_latest_vitals_lock = Lock()

# Last valid vitals preserved across packets (only updated when new values are valid)
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

# ── Trạng thái luồng Fall_Raw (pause health emits while processing) ────────
_waiting_for_fall_raw = False
_fall_raw_state_lock = Lock()      # Bảo vệ trạng thái


def _emit_with_delay(event: str, data: dict, delay_ms: int = WS_HEALTH_EMIT_DELAY_MS):
    """Emit Socket.IO event with delay to allow frontend rendering.
    
    Throttles rapid updates using a timer. If a new emit is scheduled while
    one is pending, the old one is cancelled.
    
    Args:
        event: Event name (e.g., 'health_update')
        data: Event payload dict
        delay_ms: Delay in milliseconds before emit (default 500ms)
    """
    global _emit_timer
    
    def do_emit():
        try:
            socketio.emit(event, data)
            print(f"📡 [EMIT] {event} after {delay_ms}ms delay")
        except Exception as e:
            print(f"❌ Emit error: {e}")
    
    with _emit_timer_lock:
        if _emit_timer is not None:
            _emit_timer.cancel()

        if delay_ms <= 0:
            do_emit()
            _emit_timer = None
            return

        _emit_timer = Timer(delay_ms / 1000.0, do_emit)
        _emit_timer.daemon = True
        _emit_timer.start()


def _get_fall_model():
    global _fall_model
    if _fall_model is None:
        with _fall_model_lock:
            if _fall_model is None:
                _fall_model = create_fall_model()
                print("🤖 Fall model da nap: AI(model.h5)")
    return _fall_model


def _append_history(packet):
    """Luu moi ban tin do duoc vao file lich su JSONL."""
    history_line = {
        "saved_at": int(time.time()),
        **packet,
    }
    with _history_lock:
        with open(HISTORY_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(history_line, ensure_ascii=False) + "\n")

def _handle_alert_over_3_batches(health_samples):

    global _last_alert_time
    global _buzzer_active
    global _wait_next_normal_batch

    if not health_samples:
        return

    last_bpm = None
    last_spo2 = None
    last_temp = None

    # lấy sample cuối batch
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

        # chưa đủ 3 batch
        if len(_vital_batch_buffer) < 3:
            return

        b1 = _vital_batch_buffer[0]
        b2 = _vital_batch_buffer[1]
        b3 = _vital_batch_buffer[2]   # latest

    # =====================================================
    # Nếu batch mới nhất NORMAL hoàn toàn -> tắt còi
    # =====================================================

    latest_normal = (
        not b3["bpm"] and
        not b3["spo2"] and
        not b3["temp"]
    )

    if latest_normal:

        if _buzzer_active:

            print("✅ Batch mới nhất NORMAL -> tắt còi")

            _stop_buzzer()

        return

    # =====================================================
    # Nếu đang kêu rồi -> không trigger lại
    # =====================================================

    if _buzzer_active:
        return

    triggered = []

    # =====================================================
    # BPM
    # latest abnormal
    # và 1 trong 2 batch trước abnormal
    # =====================================================

    if b3["bpm"] and (b1["bpm"] or b2["bpm"]):
        triggered.append("bpm")

    # =====================================================
    # SpO2
    # =====================================================

    if b3["spo2"] and (b1["spo2"] or b2["spo2"]):
        triggered.append("spo2")

    # =====================================================
    # TEMP
    # =====================================================

    if b3["temp"] and (b1["temp"] or b2["temp"]):
        triggered.append("temp")

    if not triggered:
        return

    now = time.time()

    # cooldown chống spam
    if now - _last_alert_time < ALERT_COOLDOWN_S:
        return

    _last_alert_time = now

    print(
        f"🚨 ALERT 3-BATCH: {triggered}"
    )

    _buzzer_active = True

    _fire_beep_pattern()
def _read_history(limit=50):
    """Doc lich su do duoc tu file, uu tien ban ghi moi nhat."""
    if limit <= 0:
        return []

    if not os.path.exists(HISTORY_FILE):
        return []

    with _history_lock:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            lines = file.readlines()

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
    """
    Decode binary fall_raw payload từ ESP32 v6 (symmetric window).
    
    Format:
      - 4 bytes: uint32 BE - trigger_ts (ms khi peak được detect)
      - 2 bytes: uint16 BE - num_samples (150 hoặc 75)
      - 1 byte: uint8 - pre_samples (100 hoặc 60, chỉ số của peak trong window)
      - 1 byte: uint8 - reason length (N)
      - N bytes: reason string (trigger trigger lý do)
      - M*16 bytes: samples (M = num_samples, mỗi sample = 8*int16)
           channels: ax, ay, az, gx, gy, gz, magnitude, jerk
    
    Returns:
      dict với: trigger_ts, num_samples, pre_samples, reason, data (np.ndarray)
      hoặc None nếu decode fail
    """
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
        
        # Decode samples: 8 channels × int16 per sample = 16 bytes/sample
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
            
            samples[i, 0] = raw[0] / SCALE_ACC    # ax
            samples[i, 1] = raw[1] / SCALE_ACC    # ay
            samples[i, 2] = raw[2] / SCALE_ACC    # az
            samples[i, 3] = raw[3] / SCALE_GYRO   # gx
            samples[i, 4] = raw[4] / SCALE_GYRO   # gy
            samples[i, 5] = raw[5] / SCALE_GYRO   # gz
            samples[i, 6] = raw[6] / SCALE_MAG    # magnitude G
            samples[i, 7] = raw[7] / SCALE_JERK   # jerk
        
        print(f"✅ Decode binary: {num_samples} samples × 8 channels | peak_idx={pre_samples} | reason='{reason}'")
        
        return {
            'trigger_ts': ts,
            'num_samples': num_samples,
            'pre_samples': pre_samples,  # Peak index trong window
            'reason': reason,
            'data': samples  # shape (num_samples, 8)
        }
    
    except Exception as e:
        print(f"❌ Lỗi decode binary fall_raw: {e}")
        traceback.print_exc()
        return None



def _process_fall_raw_with_model(decoded_data: dict, alert_data: dict | None = None) -> dict | None:
 
    try:
        model = _get_fall_model()
        samples_array = decoded_data['data']      # shape (75, 8) từ fall_raw
        num_samples = decoded_data['num_samples']
        reason = decoded_data['reason']
        trigger_ts = decoded_data['trigger_ts']
        pre_samples = decoded_data['pre_samples']
        
        # ── Use fall_raw payload directly (device sends full 150 samples) ─────────
        if samples_array.shape[0] < model._required_window:
            print(
                f"⏳ fall_raw chua du: {samples_array.shape[0]}/{model._required_window} samples"
            )
            return None
        if samples_array.shape[0] > model._required_window:
            samples_array = samples_array[:model._required_window]

        motion_buf = samples_array
        print(
            f"✅ fall_raw du: {motion_buf.shape[0]}/{model._required_window} samples → Chay model"
        )
        
        # ── Get latest vitals từ sensor/data ──────────────────────────────────────
        with _latest_vitals_lock:
            hr = _latest_hr
            spo2 = _latest_spo2
        
        temp = _get_current_temperature()
        print(f"💚 Latest vitals từ sensor/data: HR={hr} BPM, SpO2={spo2:.1f}%, Temp={temp:.1f}°C")
        
        # ── Build 11-feature vector từ motion (150) + latest vitals ──────────────
        features_array = np.zeros((motion_buf.shape[0], 11), dtype=np.float32)
        
        for i in range(motion_buf.shape[0]):
            ax, ay, az = motion_buf[i, 0], motion_buf[i, 1], motion_buf[i, 2]
            gx, gy, gz = motion_buf[i, 3], motion_buf[i, 4], motion_buf[i, 5]
            
            # Tính roll, pitch từ acceleration
            roll = np.arctan2(ay, az + 1e-6)
            pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + 1e-6)
            
            # 11 features: [ax, ay, az, gx, gy, gz, roll, pitch, hr, spo2, temp]
            features_array[i] = [ax, ay, az, gx, gy, gz, roll, pitch, hr, spo2, temp]
        
        # Normalize: (X - mean) / (std + eps)
        mean = model._mean
        std = model._std
        normalized = (features_array - mean) / (std + 1e-6)
        
        # Reshape để inference: (1, num_timesteps, 11)
        X = normalized[np.newaxis, ...]  # (1, 150, 11)
        
        # Inference
        pred = model._model.predict(X, verbose=0)  # output shape: (1, 1)
        confidence = float(pred[0][0])
        
        # Threshold từ model (mặc định 0.6)
        threshold = float(os.getenv("AI_FALL_THRESHOLD", "0.6"))
        detected = confidence >= threshold
        
        print(f"🧠 [MODEL] Features=11 (150 motion) | Inference: confidence={confidence:.3f}, threshold={threshold} → detected={detected}")
        
        alert_trigger = "fall_raw"

        if alert_data:
            alert_trigger = alert_data.get("trigger",
        alert_data.get("message", "unknown"))
        
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

    # Support new payload format: {"fs": <hz>, "temp": <degC>, "data": [{"t":..., "ir":..., "red":...}, ...]}
    if isinstance(raw_payload.get("data"), list) and raw_payload.get("data"):
        series = raw_payload.get("data")
        # extract timestamps and series values
        times = []
        ir_vals = []
        red_vals = []
        for item in series:
            if not isinstance(item, dict):
                continue
            # accept keys: t, ts, timestamp
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

        # Determine ts0 and ts from times if available
        ts0_int = None
        ts_int = None
        valid_times = [t for t in times if isinstance(t, int)]
        if valid_times:
            ts0_int = valid_times[0]
            ts_int = valid_times[-1]

        # sample interval from fs (frequency Hz) if provided
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

        # If data timestamps exist, infer interval from median diff of timestamps
        sample_interval_ms_time = None
        try:
            if len(valid_times) > 1:
                import numpy as _np

                diffs = _np.diff(_np.array(valid_times, dtype=np.int64))
                # ignore non-positive diffs
                diffs = diffs[diffs > 0]
                if diffs.size > 0:
                    sample_interval_ms_time = int(max(1, int(_np.median(diffs))))
        except Exception:
            sample_interval_ms_time = None

        # Prefer timestamp-derived interval when available (more reliable), but warn if differs from fs
        if sample_interval_ms_time is not None:
            sample_interval_ms = sample_interval_ms_time
            if fs_interval_ms is not None and fs_interval_ms > 0:
                # if difference >20% warn
                diff_frac = abs(sample_interval_ms_time - fs_interval_ms) / float(fs_interval_ms)
                if diff_frac > 0.2:
                    print(f"⚠️  fs ({fs_interval_ms}ms) and timestamps-derived interval ({sample_interval_ms_time}ms) differ by {diff_frac*100:.0f}% - using timestamps")
        else:
            sample_interval_ms = fs_interval_ms

        # Build a normalized payload compatible with existing logic
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
            # Motion fields missing in this payload shape -> set to None to avoid batch-list errors
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
            len(ir_series),
            len(red_series),
            len(ax_series),
            len(ay_series),
            len(az_series),
            len(gx_series),
            len(gy_series),
            len(gz_series),
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
        # Convert empty-series -> None, single-value lists -> scalar, keep lists for true batch
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
    """Lấy nhiệt độ hiện tại (từ lần đo cuối từ sensor/data)."""
    global _current_temperature
    with _temp_lock:
        return _current_temperature


def _update_temperature(temp_value):
    """Cập nhật nhiệt độ hiện tại từ sensor/data."""
    global _current_temperature
    numeric = _to_non_negative_number(temp_value)
    if numeric is not None and 30.0 < numeric < 42.0:
        with _temp_lock:
            _current_temperature = numeric


def _build_display_payload(normalized_data, fall_prediction):
    accel_x = _to_number(normalized_data.get("ax")) or 0.0
    accel_y = _to_number(normalized_data.get("ay")) or 0.0
    accel_z = _to_number(normalized_data.get("az")) or 0.0
    gyro_x = _to_number(normalized_data.get("gx")) or 0.0
    gyro_y = _to_number(normalized_data.get("gy")) or 0.0
    gyro_z = _to_number(normalized_data.get("gz")) or 0.0

    accel_mag = (accel_x ** 2 + accel_y ** 2 + accel_z ** 2) ** 0.5
    gyro_mag = (gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2) ** 0.5

    return {
        "timestamp": normalized_data.get("ts", normalized_data.get("timestamp")),
        "status": normalized_data.get("status"),
        "vitals": {
            "bpm": normalized_data.get("bpm"),
            "spo2": normalized_data.get("spo2"),
            "temp": normalized_data.get("temp"),
        },
        "motion": {
            "accel_magnitude": round(accel_mag, 4),
            "gyro_magnitude": round(gyro_mag, 4),
        },
        "fall": {
            "detected": fall_prediction.get("fall_detected", False),
            "label": fall_prediction.get("label"),
            "state": fall_prediction.get("state"),
            "confidence_percent": fall_prediction.get("confidence_percent"),
        },
    }


def _build_health_update_with_fall(fall_result: dict) -> dict | None:

    with _packet_lock:
        if _latest_packet is None:
            print("⚠️ Không có health vitals")
            return None

        latest = _latest_packet.copy()

        # ===== FALL ROOT LEVEL =====
        latest["fall"] = {
            "detected": fall_result.get("detected", False),
            "confidence": fall_result.get("confidence", 0.0),
        }

        # ===== STATUS =====
        if fall_result.get("detected"):
            latest["data"]["status"] = ["FALL_DETECTED"]

        latest["server_timestamp"] = int(time.time())

        return latest

def _to_flutter_packet(health_data, source_topic):
    """Build health_update packet from health_data.
    
    NOTE: Fall detection is NOT run on sensor/data (luồng 3) because:
    - sensor/data only has 50 IR/Red samples (no motion data)
    - Fall detection requires IMU motion data (ax, ay, az, gx, gy, gz)
    - Fall detection only runs on sensor/fall_raw (luồng 2) with full 75×8 samples
    - Therefore fall is always {detected: False, confidence: 0.0} here
    """
    raw_data = health_data.to_dict()
    data = _normalize_health_data_for_frontend(raw_data)
    
    packet = {
        "type": "health_update",
        "source_topic": source_topic,
        "server_timestamp": int(time.time()),
        "data": data,
        "fall": {"detected": False, "confidence": 0.0},  # No motion data on sensor/data
    }

    if API_VERBOSE_OUTPUT:
        packet["data_raw"] = raw_data
        packet["display"] = _build_display_payload(raw_data, {})

    return packet

# ====== CALLBACK ======
def on_connect(client, userdata, flags, reason_code, properties=None):
    global _mqtt_connected
    # reason_code == 0 => connection successful
    if reason_code == 0:
        # Only print the connected message on transition from disconnected -> connected
        if not _mqtt_connected:
            print("✅ Da ket noi HiveMQ Cloud")
        _mqtt_connected = True

        # Ensure subscriptions are (re)registered on every connect
        client.subscribe("sensor/data")
        client.subscribe("sensor/fall_raw")
        client.subscribe("health/#")

        if not _mqtt_connected:
            print("📡 Da dang ky topic: sensor/#, health/#")
    else:
        print("❌ Ket noi that bai, ma loi:", reason_code)


def on_disconnect(client, userdata, rc, properties=None):
    """MQTT disconnect handler: mark disconnected and log reason."""
    global _mqtt_connected
    _mqtt_connected = False
    print(f"⚠️ MQTT disconnected, rc={rc}")


def on_message(client, userdata, msg):
    print("\n📩 ===== TIN NHAN MOI =====")
    print("📌 Chu de:", msg.topic)

    try:
        global _latest_packet
        global _latest_raw_payload
        global _waiting_for_fall_raw
        global _latest_hr
        global _latest_spo2
        global _last_alert_status

        # NOTE: `sensor/alert` handling removed — fall processing triggers on `sensor/fall_raw` only.

        # ════════════════════════════════════════════════════════════════════════════════
        # [LUỒNG 2B] SENSOR FALL_RAW - Dữ liệu motion sau alert (detect ngã)
        # ════════════════════════════════════════════════════════════════════════════════
        if msg.topic == "sensor/fall_raw":
            with _fall_raw_state_lock:
                # When fall_raw arrives we pause health emits and run the fall model.
                # We no longer require a prior alert to trigger processing.
                _waiting_for_fall_raw = True
                pending_alert = None

            try:
                # Decode binary payload (75 hoặc 150 samples × 8 channels)
                decoded = _decode_fall_raw_binary(msg.payload)
                if not decoded:
                    print("❌ Không thể decode fall_raw binary")
                    return

                # Pause health updates while fall model is running (default 5s).
                if FALL_MODEL_BLOCK_MS > 0:
                    print(f"⏸️  Tạm dừng {FALL_MODEL_BLOCK_MS}ms để xử lý fall model")
                    time.sleep(FALL_MODEL_BLOCK_MS / 1000.0)

                # ────────────────────────────────────────────────────────────────────────
                # Motion data (150 mẫu) → Build 11-feature matrix → Model inference
                # ────────────────────────────────────────────────────────────────────────
                print(f"🧠 [MODEL] Đang xử lý {decoded['num_samples']} motion samples...")
                result = _process_fall_raw_with_model(decoded, pending_alert)

                if result:
                    if result['detected']:
                        print(f"🚨 [RESULT] ✅ TẾ NGÃ XÁC NHẬN! | Confidence: {result['confidence']:.3f}")
                    else:
                        print(f"✓ [RESULT] Không phải ngã | Confidence: {result['confidence']:.3f}")

                    # ────────────────────────────────────────────────────────────────
                    # Kết hợp fall detection result với health vitals cuối cùng
                    # → Emit SINGLE "health_update" event (không tách thành 2 responses)
                    # ────────────────────────────────────────────────────────────────
                    health_with_fall = _build_health_update_with_fall(result)

                    if health_with_fall:
                        # Persist latest packet so API can return this combined result
                        with _packet_lock:
                            _latest_packet = health_with_fall
                            # leave _latest_raw_payload unchanged (no raw sensor/data for fall_raw)

                        # 💓 Gửi MỘT response duy nhất: health_update (kết hợp fall + vitals)
                        _append_history(health_with_fall)
                        _emit_with_delay("health_update", health_with_fall, delay_ms=WS_FALL_EMIT_DELAY_MS)

                        fall_status = "TE_NGA" if result['detected'] else "NO_FALL"
                        print(f"💓 [RESPONSE] {fall_status} | BPM={health_with_fall['data'].get('bpm')} "
                              f"SpO2={health_with_fall['data'].get('spo2')}% Temp={health_with_fall['data'].get('temp')}°C")

                        # Nếu ngã được xác nhận → kích hoạt cảnh báo trên thiết bị
                        if result['detected']:
                            _send_device_command("buzzer_on", duration_ms=5000)
                    else:
                        print("⚠️  Không thể kết hợp fall detection với health vitals")
            finally:
                with _fall_raw_state_lock:
                    _waiting_for_fall_raw = False

            return

        # ════════════════════════════════════════════════════════════════════════════════
        # [LUỒNG 1] SENSOR DATA - Đọc dữ liệu sức khỏe (SpO2/BPM) thường xuyên từ ESP32
        # ════════════════════════════════════════════════════════════════════════════════
        
        # Optional input delay for noisy deployments; default is 0 for low latency.
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
        
        # Cập nhật temperature hiện tại từ sensor/data
        _update_temperature(raw_payload.get("temp"))
        
        health_samples = from_json_samples(json.dumps(normalized_raw_payload))

        latest_packet = None
        for health_data in health_samples:
            # Classify health status based on vitals
            # Use last-valid vitals as fallback for classification when current sample missing/zero
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
            
            packet = _to_flutter_packet(health_data, msg.topic)
            
            # Update status in the packet
            packet["data"]["status"] = status
            latest_packet = packet

            # Send buzzer pattern whenever a bad status appears
      
            # Update preserved last-valid vitals after building packet
            with _latest_vitals_lock:
                if _is_valid_vital_value("heart_rate", health_data.heart_rate):
                    _last_valid_vitals["heart_rate"] = float(health_data.heart_rate)
                if _is_valid_vital_value("spo2", health_data.spo2):
                    _last_valid_vitals["spo2"] = float(health_data.spo2)
                # temp update via specialized function, but also preserve here
                if _is_valid_vital_value("temp", health_data.temp):
                    _last_valid_vitals["temp"] = float(health_data.temp)
        # _handle_alert_for_batch(health_samples)
        _handle_alert_over_3_batches(health_samples)
        # _update_vital_windows(health_samples)
        # _check_window_alerts()
        if latest_packet is not None:
            with _packet_lock:
                # If incoming packet reports BPM==0 or None, preserve previous valid BPM
                try:
                    bpm_val = latest_packet.get("data", {}).get("bpm")
                except Exception:
                    bpm_val = None

                if bpm_val is None or bpm_val == 0:
                    if _latest_hr is not None and _latest_hr > 0:
                        latest_packet["data"]["bpm"] = _latest_hr
                        bpm_val = _latest_hr

                # Update stored latest vitals (only when meaningful)
                if bpm_val is not None:
                    try:
                        _latest_hr = float(bpm_val)
                    except Exception:
                        pass

                try:
                    spo2_val = latest_packet.get("data", {}).get("spo2")
                except Exception:
                    spo2_val = None

                if spo2_val is None:
                    if _latest_spo2 is not None and _latest_spo2 > 0:
                        latest_packet["data"]["spo2"] = _latest_spo2
                        spo2_val = _latest_spo2

                if spo2_val is not None:
                    try:
                        _latest_spo2 = float(spo2_val)
                    except Exception:
                        pass

                _latest_packet = latest_packet
                _latest_raw_payload = normalized_raw_payload

            with _fall_raw_state_lock:
                waiting_for_fall_raw = _waiting_for_fall_raw

            if waiting_for_fall_raw:
                print(
                    "⏳ [LUỒNG 1 - SENSOR DATA] Dang cho fall_raw, tam ngung emit health_update"
                )
            else:
                # Save and push only the final packet for each MQTT message.
                _append_history(latest_packet)
                _emit_with_delay("health_update", latest_packet, delay_ms=WS_HEALTH_EMIT_DELAY_MS)

                print(
                    f"💓 [LUỒNG 1 - SENSOR DATA] Đã xử lý {len(health_samples)} mẫu từ {msg.topic} | "
                    f"BPM={latest_packet['data']['bpm']} SpO2={latest_packet['data']['spo2']}% "
                    f"Temp={latest_packet['data']['temp']}°C"
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
    """Send command to device via MQTT device/command topic."""
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
    """Kiểm tra xem một vital có vượt ngưỡng không.
 
    Returns True nếu value < LOW hoặc value > HIGH.
    Returns False nếu value là None (không đủ dữ liệu → không trigger).
    """
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
    """Huỷ tất cả timer buzzer đang pending."""
    with _alert_timers_lock:
        for t in _alert_timers:
            try:
                t.cancel()
            except Exception:
                pass
        _alert_timers.clear()
 
 
def _fire_beep_pattern():
    """
    Gửi chuỗi lệnh buzzer_on theo tần số trong ALERT_BEEP_DURATION_S giây.
    Mỗi beep = buzzer_on(duration=ALERT_BEEP_ON_MS), cách nhau ALERT_BEEP_OFF_MS.
    Toàn bộ pattern kéo dài ~ALERT_BEEP_DURATION_S giây.
    """
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


# NOTE: `/api/health/latest` endpoint removed by request. Use `/api/health/history` to fetch recent records.


@app.get("/api/docs")
def get_api_docs():
    return jsonify(
        {
            "name": "Heart Rate Monitor Backend API",
            "version": "1.0",
            "base_url_hint": "/api",
            "http_endpoints": [
                {
                    "method": "GET",
                    "path": "/api/docs",
                    "description": "Tai lieu API cho frontend.",
                },
                {
                    "method": "GET",
                    "path": "/api/health/history",
                    "description": "Lay lich su health_update moi nhat.",
                    "query": {
                        "limit": {
                            "type": "integer",
                            "min": 1,
                            "max": 1000,
                            "default": 50,
                        }
                    },
                },
            ],
            "websocket": {
                "transport": "Socket.IO",
                "events": [
                    {
                        "event": "health_update",
                        "description": "Ban tin tong hop suc khoe (BPM/SpO2/Temp/Status) va co the kem ket qua te nga.",
                        "payload_shape": {
                            "type": "health_update",
                            "source_topic": "sensor/data",
                            "timestamp": "unix_seconds",
                            "data": {
                                "bpm": "number",
                                "spo2": "number_or_null",
                                "temp": "number",
                                "status": "NORMAL|FEVER|HIGH_HEART_RATE|LOW_HEART_RATE|FALL_DETECTED",
                                "fall": {"detected": "bool", "confidence": "number"},
                            },
                        },
                    },
                    {
                        "event": "health_error",
                        "description": "Loi parse/validate payload MQTT.",
                    },
                ],
            },
            "notes": [
                "Frontend nen uu tien nghe health_update theo thoi gian thuc.",
                "Neu can debug raw payload tu ESP32, lay ban ghi moi nhat tu /api/health/history.",
                "Gia tri BPM co the la 0 khi tin hieu IR khong du chat luong de detect peak.",
            ],
        }
    )


@app.get("/api/health/history")
def get_health_history():
    raw_limit = request.args.get("limit", "50")
    try:
        limit = int(raw_limit)
    except ValueError:
        return jsonify({"message": "Tham so 'limit' phai la so nguyen"}), 400

    if limit < 1 or limit > 1000:
        return jsonify({"message": "Tham so 'limit' phai trong khoang 1..1000"}), 400

    history = _read_history(limit=limit)
    return jsonify({
        "count": len(history),
        "items": history,
    })
def main():
    global _mqtt_client
    
    _mqtt_client = build_mqtt_client()
    mqtt_started = False

    try:
        try:
            # Connect asynchronously so API can bind port immediately on cloud deploy.
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
        print("🔌 Su kien WebSocket: health_update")
        # Render temp deploy: allow Werkzeug server to run in non-debug env.
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