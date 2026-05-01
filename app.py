import paho.mqtt.client as mqtt
import ssl
import certifi
import time
import os
import json
import traceback
from threading import Lock

from flask import Flask, jsonify, request
from flask_socketio import SocketIO

from fall_model import FallInput, create_fall_model
from model import from_json_samples, classify

BROKER = "dfee921e5f16440e8f3892ed3564c06d.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
USERNAME = "hungdaica123"
PASSWORD = "Hungpro123"
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


def _get_fall_model():
    global _fall_model
    if _fall_model is None:
        with _fall_model_lock:
            if _fall_model is None:
                _fall_model = create_fall_model(use_ai=USE_AI_FALL_MODEL)
                print(f"🤖 Fall model da nap: {'AI(model.h5)' if USE_AI_FALL_MODEL else 'RuleBased'}")
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


def _normalize_raw_payload_for_api(raw_payload):
    if not isinstance(raw_payload, dict):
        raise ValueError("Payload khong hop le, can doi tuong JSON")

    ts = raw_payload.get("ts", raw_payload.get("timestamp"))
    ts_int = int(ts) if ts is not None else int(time.time() * 1000)

    temp = _to_non_negative_number(raw_payload.get("temp"))
    return {
        "ts": ts_int,
        "ts0": raw_payload.get("ts0"),
        "temp": round(temp, 2) if temp is not None else None,
        "sample_interval_ms": raw_payload.get("sample_interval_ms"),
        "ir": _normalize_series_field(raw_payload.get("ir"), "ir", as_int=True),
        "red": _normalize_series_field(raw_payload.get("red"), "red", as_int=True),
        "ax": _normalize_series_field(raw_payload.get("ax"), "ax"),
        "ay": _normalize_series_field(raw_payload.get("ay"), "ay"),
        "az": _normalize_series_field(raw_payload.get("az"), "az"),
        "gx": _normalize_series_field(raw_payload.get("gx"), "gx"),
        "gy": _normalize_series_field(raw_payload.get("gy"), "gy"),
        "gz": _normalize_series_field(raw_payload.get("gz"), "gz"),
    }


def _is_fall_alert_payload(raw_payload, topic):
    if isinstance(topic, str) and topic.endswith("/alert"):
        return True
    if not isinstance(raw_payload, dict):
        return False
    return raw_payload.get("type") == "fall_alert" or raw_payload.get("event") == "fall_alert"


def _build_fall_alert_packet(raw_payload, source_topic):
    total_g = _to_number(raw_payload.get("total_g"))
    ts = raw_payload.get("ts", raw_payload.get("timestamp"))

    return {
        "type": "fall_alert",
        "source_topic": source_topic,
        "server_timestamp": int(time.time()),
        "data": {
            "ts": int(ts) if ts is not None else None,
            "total_g": round(total_g, 2) if total_g is not None else None,
            "message": raw_payload.get("message", "fall_detected"),
        },
    }


def _is_lstm_fall_raw_payload(raw_payload, topic):
    """Kiểm tra xem payload có phải từ sensor/fall_raw (dữ liệu AI LSTM) không."""
    if not isinstance(topic, str) or not isinstance(raw_payload, dict):
        return False
    if topic == "sensor/fall_raw":
        return True
    if raw_payload.get("event") == "fall_raw":
        return True
    return False


def _process_lstm_raw_data(raw_payload, source_topic):
    """Xử lý dữ liệu AI LSTM từ sensor/fall_raw (200 mẫu, 4 giây).
    
    Payload format:
    {
        "event": "fall_raw",
        "trigger_ts": ...,
        "ax": [val1, val2, ..., val200],
        "ay": [...], "az": [...],
        "gx": [...], "gy": [...], "gz": [...]
    }
    """
    try:
        model = _get_fall_model()
        temp = _get_current_temperature()  # Lấy temp từ lần đo cuối
        trigger_ts = raw_payload.get("trigger_ts", int(time.time() * 1000))
        
        # Kiểm tra xem có đủ dữ liệu motion không
        required_keys = ["ax", "ay", "az", "gx", "gy", "gz"]
        for key in required_keys:
            if key not in raw_payload or not isinstance(raw_payload[key], list):
                print(f"⚠️  Dữ liệu LSTM thiếu: {key}")
                return None
        
        sample_count = len(raw_payload["ax"])
        
        # Kiểm tra tất cả mảng có độ dài bằng nhau không
        for key in required_keys[1:]:
            if len(raw_payload[key]) != sample_count:
                print(f"⚠️  Dữ liệu LSTM không đồng nhất: {key} có {len(raw_payload[key])} mẫu, mong đợi {sample_count}")
                return None
        
        # Xử lý từng mẫu thông qua model
        fall_detected = False
        max_confidence = 0.0
        
        print(f"🧠 [AI LSTM] Đang xử lý {sample_count} mẫu dữ liệu từ {source_topic}...")
        
        for idx in range(sample_count):
            fall_input = FallInput(
                ax=float(raw_payload["ax"][idx]),
                ay=float(raw_payload["ay"][idx]),
                az=float(raw_payload["az"][idx]),
                gx=float(raw_payload["gx"][idx]),
                gy=float(raw_payload["gy"][idx]),
                gz=float(raw_payload["gz"][idx]),
                heart_rate=0.0,  # Không có BPM từ LSTM data
                spo2=0.0,        # Không có SpO2 từ LSTM data
                temp=temp,       # Dùng temp hiện tại
                timestamp=trigger_ts + idx * 20,  # 20ms interval (50Hz)
            )
            
            prediction = model.predict(fall_input).to_dict()
            if prediction.get("detected", False):
                fall_detected = True
                confidence = prediction.get("confidence", 0.0)
                max_confidence = max(max_confidence, confidence)
        
        return {
            "detected": fall_detected,
            "confidence": max_confidence,
            "sample_count": sample_count,
            "trigger_ts": trigger_ts,
        }
    
    except Exception as exc:
        print(f"❌ Lỗi xử lý dữ liệu LSTM: {exc}")
        return None


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


def _format_fall_for_frontend(fall_prediction):
    detected = bool(fall_prediction.get("detected", False))
    confidence = _to_number(fall_prediction.get("confidence"))
    confidence = 0.0 if confidence is None else max(0.0, min(1.0, confidence))
    
    return {
        "detected": detected,
        "confidence": round(confidence, 3),
    }


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


def _to_flutter_packet(health_data, source_topic):
    raw_data = health_data.to_dict()
    data = _normalize_health_data_for_frontend(raw_data)
    
    # Only run fall detection on sensor/fall_raw (has motion data)
    # sensor/data has no motion data, so fall should be false
    fall_prediction = {"detected": False, "confidence": 0.0}
    
    if source_topic == "sensor/fall_raw":
        model = _get_fall_model()
        fall_input = FallInput(
            ax=health_data.ax,
            ay=health_data.ay,
            az=health_data.az,
            gx=health_data.gx,
            gy=health_data.gy,
            gz=health_data.gz,
            heart_rate=health_data.heart_rate,
            spo2=health_data.spo2 or 0.0,
            temp=health_data.temp,
            timestamp=health_data.timestamp,
        )
        fall_prediction = _format_fall_for_frontend(model.predict(fall_input).to_dict())
    
    packet = {
        "type": "health_update",
        "source_topic": source_topic,
        "server_timestamp": int(time.time()),
        "data": data,
        "fall": fall_prediction,
    }

    if API_VERBOSE_OUTPUT:
        packet["data_raw"] = raw_data
        packet["display"] = _build_display_payload(raw_data, fall_prediction)

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

        raw = msg.payload.decode()
        try:
            raw_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Payload JSON khong hop le") from exc

        # ────────────────────────────────────────────────────────────────
        # [LUỒNG 1] SENSOR ALERT - Cảnh báo tức thời
        # ────────────────────────────────────────────────────────────────
        if _is_fall_alert_payload(raw_payload, msg.topic):
            alert_packet = _build_fall_alert_packet(raw_payload, msg.topic)
            with _packet_lock:
                _latest_raw_payload = raw_payload

            _append_history(alert_packet)
            socketio.emit("sensor_alert", alert_packet)
            print(
                f"🚨 [LUỒNG 1 - ALERT] Da nhan alert te nga tu {msg.topic} | total_g={alert_packet['data']['total_g']}"
            )
            return

        # ────────────────────────────────────────────────────────────────
        # [LUỒNG 2] SENSOR FALL_RAW - Dữ liệu AI LSTM (200 mẫu, 4 giây)
        # ────────────────────────────────────────────────────────────────
        if _is_lstm_fall_raw_payload(raw_payload, msg.topic):
            print(f"🧠 [LUỒNG 2 - AI LSTM] Nhan du lieu tu {msg.topic}")
            
            lstm_result = _process_lstm_raw_data(raw_payload, msg.topic)
            if lstm_result:
                # Tạo packet AI dựa trên kết quả LSTM
                ai_packet = {
                    "type": "fall_lstm_result",
                    "source_topic": msg.topic,
                    "server_timestamp": int(time.time()),
                    "data": {
                        "detected": lstm_result["detected"],
                        "confidence": round(lstm_result["confidence"], 3),
                        "sample_count": lstm_result["sample_count"],
                    },
                }
                
                with _packet_lock:
                    _latest_raw_payload = raw_payload

                _append_history(ai_packet)
                socketio.emit("ai_fall_result", ai_packet)
                
                if lstm_result["detected"]:
                    print(f"🚨 [AI] Xác nhận TE NGA! Confidence: {lstm_result['confidence']:.3f}")
                    _send_device_command("buzzer_on", duration_ms=5000)
                else:
                    print(f"✓ [AI] Không phải vấp ngã. Confidence: {lstm_result['confidence']:.3f}")
            return

        # ────────────────────────────────────────────────────────────────
        # [LUỒNG 3] SENSOR DATA - Dữ liệu SpO2/BPM thường xuyên (1Hz)
        # ────────────────────────────────────────────────────────────────
        normalized_raw_payload = _normalize_raw_payload_for_api(raw_payload)
        
        # Cập nhật temperature hiện tại từ sensor/data
        _update_temperature(raw_payload.get("temp"))
        
        health_samples = from_json_samples(raw)

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

            # Save and push only the final packet for each MQTT message.
            _append_history(latest_packet)
            socketio.emit("health_update", latest_packet)

            print(
                f"💓 [LUỒNG 3 - SENSOR] Da xu ly {len(health_samples)} mau tu {msg.topic} | "
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