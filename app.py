import paho.mqtt.client as mqtt
import ssl
import certifi
import time
import os
import json
import traceback
import struct
from threading import Lock, Timer

import numpy as np

from flask import Flask, jsonify, request
from flask_socketio import SocketIO

from fall_model import create_fall_model
from model import from_json_samples, classify, STATUS_FALL_DETECTED

BROKER = "11060dbd13b54fc988ae8f9bfc43c089.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
USERNAME = "heart-rate"
PASSWORD = "aB123456"
CLIENT_ID = "python_backend"
MQTT_REQUIRED = os.getenv("MQTT_REQUIRED", "false").lower() == "true"
API_BIND_HOST = os.getenv("API_BIND_HOST", "0.0.0.0")
API_ACCESS_HOST = os.getenv("API_ACCESS_HOST", "127.0.0.1")
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
_current_temperature = 36.5  # Nhiệt độ cơ thể mặc định (lấy từ sensor/data)
_temp_lock = Lock()
_emit_timer = None  # Throttle timer cho Socket.IO emit
_emit_timer_lock = Lock()

# ── Lưu latest vitals từ sensor/data (chỉ 1 value) ────────────────────────────
_latest_hr = 0.0          # Latest heart rate (BPM) từ sensor/data
_latest_spo2 = 0.0        # Latest SpO2 (%) từ sensor/data
_latest_vitals_lock = Lock()

# ── Trạng thái luồng Alert → Fall_Raw ────────────────────────────────────────
_waiting_for_fall_raw = False      # Chờ dữ liệu fall_raw sau alert
_pending_alert_data = None         # Lưu alert data để match với fall_raw
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



def _process_fall_raw_with_model(decoded_data: dict, alert_data: dict) -> dict | None:
 
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
        
        return {
            'detected': detected,
            'confidence': round(confidence, 3),
            'num_samples': num_samples,
            'pre_samples': pre_samples,
            'trigger_ts': trigger_ts,
            'reason': reason,
            'alert_trigger': alert_data.get("trigger", alert_data.get("message", "unknown")),
        }
    
    except Exception as e:
        print(f"❌ Lỗi xử lý fall_raw qua model: {e}")
        traceback.print_exc()
        return None


def _normalize_raw_payload_for_api(raw_payload):
    if not isinstance(raw_payload, dict):
        raise ValueError("Payload khong hop le, can doi tuong JSON")

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
        "ax": ax_series,
        "ay": ay_series,
        "az": az_series,
        "gx": gx_series,
        "gy": gy_series,
        "gz": gz_series,
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
    """
    Tạo health_update packet kết hợp fall detection result + health vitals cuối cùng.
    Đảm bảo fall detection được ghi kèm theo tình trạng sức khỏe.
    
    Hòa nhập:
    - Fall detection (AI model confidence)
    - Latest vitals (BPM, SpO2, Temp, Status)
    - Alert trigger reason
    
    Returns health_update packet hoặc None nếu không có vitals
    """
    with _packet_lock:
        if _latest_packet is None:
            print("⚠️  Không có health vitals để kết hợp với fall detection")
            return None
        
        # Clone packet cuối cùng
        latest = _latest_packet.copy()
        
        # Merge fall detection result vào data
        latest["data"]["fall"] = {
            "detected": fall_result.get("detected", False),
            "confidence": fall_result.get("confidence", 0.0),
            "trigger": fall_result.get("alert_trigger", "unknown"),
            "reason": fall_result.get("reason", ""),
        }
        
        # Update status nếu fall detected
        if fall_result.get("detected"):
            latest["data"]["status"] = STATUS_FALL_DETECTED
        
        # Metadata từ fall detection
        latest["fall_detection"] = {
            "timestamp": fall_result.get("trigger_ts"),
            "num_samples": fall_result.get("num_samples"),
            "pre_samples": fall_result.get("pre_samples"),
            "model_version": "LSTM_v6",
        }
        
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
    if reason_code == 0:
        print("✅ Da ket noi HiveMQ Cloud")

        client.subscribe("sensor/#")
        client.subscribe("health/#")

        print("📡 Da dang ky topic: sensor/#, health/#")

    else:
        print("❌ Ket noi that bai, ma loi:", reason_code)


def on_message(client, userdata, msg):
    print("\n📩 ===== TIN NHAN MOI =====")
    print("📌 Chu de:", msg.topic)

    try:
        global _latest_packet
        global _latest_raw_payload
        global _waiting_for_fall_raw
        global _pending_alert_data

        # ════════════════════════════════════════════════════════════════════════════════
        # [LUỒNG 2A] SENSOR ALERT - Cảnh báo tức thời (kích hoạt fall detection)
        # ════════════════════════════════════════════════════════════════════════════════
        if msg.topic == "sensor/alert":
            print("🚨 [ALERT] Nhận cảnh báo từ sensor → chờ dữ liệu fall_raw...")
            try:
                raw = msg.payload.decode()
                alert_payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"⚠️  Alert payload không phải JSON: {e}")
                return
            
            with _fall_raw_state_lock:
                _waiting_for_fall_raw = True
                _pending_alert_data = alert_payload
            
            alert_packet = {
                "type": "fall_alert",
                "source_topic": msg.topic,
                "server_timestamp": int(time.time()),
                "data": {
                    "ts": alert_payload.get("ts", alert_payload.get("trigger")),
                    "trigger": alert_payload.get("trigger", "unknown"),
                    "peak_g": alert_payload.get("peak_g"),
                    "delta_g": alert_payload.get("delta_g"),
                },
            }
            
            _append_history(alert_packet)
            socketio.emit("sensor_alert", alert_packet)
            print(f"🚨 [ALERT] Ghi lại cảnh báo | trigger={alert_payload.get('trigger')} | chờ tối đa 10 giây")
            return

        # ════════════════════════════════════════════════════════════════════════════════
        # [LUỒNG 2B] SENSOR FALL_RAW - Dữ liệu motion sau alert (detect ngã)
        # ════════════════════════════════════════════════════════════════════════════════
        if msg.topic == "sensor/fall_raw":
            with _fall_raw_state_lock:
                if not _waiting_for_fall_raw:
                    print("⚠️  Nhận fall_raw nhưng không có alert trước đó → bỏ qua")
                    return
                
                _waiting_for_fall_raw = False
                pending_alert = _pending_alert_data
                _pending_alert_data = None
            
            # Decode binary payload (75 hoặc 150 samples × 8 channels)
            decoded = _decode_fall_raw_binary(msg.payload)
            if not decoded:
                print("❌ Không thể decode fall_raw binary")
                return
            
            # ────────────────────────────────────────────────────────────────────────────
            # Motion data (150 mẫu) → Build 11-feature matrix → Model inference
            # ────────────────────────────────────────────────────────────────────────────
            print(f"🧠 [MODEL] Đang xử lý {decoded['num_samples']} motion samples...")
            result = _process_fall_raw_with_model(decoded, pending_alert)
            
            if result:
                if result['detected']:
                    print(f"🚨 [RESULT] ✅ TẾ NGÃ XÁC NHẬN! | Confidence: {result['confidence']:.3f}")
                else:
                    print(f"✓ [RESULT] Không phải ngã | Confidence: {result['confidence']:.3f}")
                
                # ────────────────────────────────────────────────────────────────────────
                # Kết hợp fall detection result với health vitals cuối cùng
                # → Emit SINGLE "health_update" event (không tách thành 2 responses)
                # ────────────────────────────────────────────────────────────────────────
                health_with_fall = _build_health_update_with_fall(result)
                
                if health_with_fall:
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
            
            return

        # ════════════════════════════════════════════════════════════════════════════════
        # [LUỒNG 1] SENSOR DATA - Đọc dữ liệu sức khỏe (SpO2/BPM) thường xuyên từ ESP32
        # ════════════════════════════════════════════════════════════════════════════════
        
        # Optional input delay for noisy deployments; default is 0 for low latency.
        if SENSOR_INPUT_PROCESS_DELAY_MS > 0:
            time.sleep(SENSOR_INPUT_PROCESS_DELAY_MS / 1000.0)
        
        raw = msg.payload.decode()
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
            status = classify(health_data.heart_rate, health_data.temp)
            
            packet = _to_flutter_packet(health_data, msg.topic)
            
            # Update status in the packet
            packet["data"]["status"] = status
            latest_packet = packet

        if latest_packet is not None:
            with _packet_lock:
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


@app.get("/api/health/latest")
def get_latest_health():
    output_format = request.args.get("format", "packet").strip().lower()

    if output_format not in ("packet", "raw"):
        return jsonify({"message": "Tham so 'format' chi chap nhan: packet, raw"}), 400

    with _packet_lock:
        if output_format == "raw":
            if _latest_raw_payload is None:
                return jsonify({"message": "Chua nhan duoc du lieu raw"}), 404
            return jsonify(_latest_raw_payload)

        if _latest_packet is None:
            return jsonify({"message": "Chua nhan duoc du lieu suc khoe"}), 404
        return jsonify(_latest_packet)


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
                    "path": "/api/health/latest",
                    "description": "Lay goi du lieu suc khoe moi nhat.",
                    "query": {
                        "format": {
                            "type": "string",
                            "allowed": ["packet", "raw"],
                            "default": "packet",
                        }
                    },
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
                        "event": "sensor_alert",
                        "description": "Canh bao checking tu luong phat hien nga truoc khi co ket qua AI.",
                    },
                    {
                        "event": "health_error",
                        "description": "Loi parse/validate payload MQTT.",
                    },
                ],
            },
            "notes": [
                "Frontend nen uu tien nghe health_update theo thoi gian thuc.",
                "Neu can debug raw payload tu ESP32, goi /api/health/latest?format=raw.",
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
        if mqtt_started:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()