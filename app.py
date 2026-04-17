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
from model import from_json_samples

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
_packet_lock = Lock()
_history_lock = Lock()
_fall_model_lock = Lock()
_fall_model = None


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


def _format_fall_for_frontend(fall_prediction):
    fall = dict(fall_prediction)
    confidence = _to_number(fall.get("confidence"))
    confidence = 0.0 if confidence is None else max(0.0, min(1.0, confidence))
    fall_detected = bool(fall.get("fall_detected", False))
    label = str(fall.get("label", "UNKNOWN"))

    if label == "WARMUP":
        state = "warming_up"
    elif fall_detected:
        state = "alert"
    else:
        state = "normal"

    fall["confidence"] = round(confidence, 3)
    fall["confidence_percent"] = round(confidence * 100.0, 1)
    fall["state"] = state
    return fall


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

        client.subscribe("health/#")

        print("📡 Da dang ky topic: health/#")

    else:
        print("❌ Ket noi that bai, ma loi:", reason_code)


def on_message(client, userdata, msg):
    print("\n📩 ===== TIN NHAN MOI =====")
    print("📌 Chu de:", msg.topic)

    try:
        raw = msg.payload.decode()
        health_samples = from_json_samples(raw)

        latest_packet = None
        for health_data in health_samples:
            packet = _to_flutter_packet(health_data, msg.topic)
            latest_packet = packet

            with _packet_lock:
                global _latest_packet
                _latest_packet = packet

            _append_history(packet)
            socketio.emit("health_update", packet)

        if latest_packet is not None:
            print(
                f"📊 Da xu ly {len(health_samples)} mau | "
                f"fall={latest_packet['fall']['label']} conf={latest_packet['fall']['confidence']}"
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
    with _packet_lock:
        if _latest_packet is None:
            return jsonify({"message": "Chua nhan duoc du lieu suc khoe"}), 404
        return jsonify(_latest_packet)


@app.get("/api/health/schema")
def get_health_schema():
    example = {
        "accepted_input_formats": {
            "single_sample": {
                "ts": 1710000000,
                "bpm": 78,
                "spo2": 97,
                "temp": 36.5,
                "ax": 0.12,
                "ay": -0.98,
                "az": 9.81,
                "gx": 0.01,
                "gy": 0.02,
                "gz": 0.0,
                "ir": 523456,
                "red": 498123,
            },
            "batch_samples": {
                "ts": 1710000000,
                "temp": 36.5,
                "sample_interval_ms": 20,
                "ir": [523456, 523501, 523520],
                "red": [498123, 498166, 498201],
                "ax": [0.12, 0.08, -0.21],
                "ay": [-0.98, -1.03, -0.88],
                "az": [9.81, 9.78, 9.74],
                "gx": [0.01, 0.03, -0.02],
                "gy": [0.02, 0.01, -0.01],
                "gz": [0.0, -0.01, 0.02],
            },
        },
        "output_item": {
            "type": "health_update",
            "source_topic": "health/data",
            "server_timestamp": 1710000001,
            "data": {
                "ts": 1710000000,
                "bpm": 78,
                "spo2": 97,
                "temp": 36.5,
                "status": "NORMAL",
            },
            "fall": {
                "fall_detected": False,
                "confidence": 0.423,
                "confidence_percent": 42.3,
                "label": "NO_FALL",
                "state": "normal",
            },
        },
        "optional_verbose_fields": {
            "enable_env": "API_VERBOSE_OUTPUT=true",
            "extra_keys": ["data_raw", "display"],
        },
    }
    return jsonify(example)


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
    client = build_mqtt_client()
    mqtt_started = False

    try:
        try:
            # Connect asynchronously so API can bind port immediately on cloud deploy.
            client.connect_async(BROKER, MQTT_PORT)
            client.loop_start()
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