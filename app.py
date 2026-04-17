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
from model import from_json

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


def _resolve_api_port() -> int:
    raw_port = os.getenv("API_PORT") or os.getenv("PORT") or "5050"
    try:
        return int(raw_port)
    except ValueError:
        return 5050


API_PORT = _resolve_api_port()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
fall_model = create_fall_model(use_ai=USE_AI_FALL_MODEL)

_latest_packet = None
_packet_lock = Lock()
_history_lock = Lock()


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


def _to_flutter_packet(health_data, source_topic):
    data = health_data.to_dict()
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
    fall_prediction = fall_model.predict(fall_input).to_dict()

    return {
        "type": "health_update",
        "source_topic": source_topic,
        "server_timestamp": int(time.time()),
        "data": data,
        "fall": fall_prediction,
    }

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
        health_data = from_json(raw)
        packet = _to_flutter_packet(health_data, msg.topic)

        with _packet_lock:
            global _latest_packet
            _latest_packet = packet

        _append_history(packet)

        print("📊 Du lieu:", packet)
        socketio.emit("health_update", packet)
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
        "type": "health_update",
        "source_topic": "health/data",
        "server_timestamp": 1710000001,
        "data": {
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
            "heart_rate": 78,
            "timestamp": 1710000000,
            "status": "NORMAL",
        },
        "fall": {
            "fall_detected": False,
            "confidence": 0.423,
            "label": "NO_FALL",
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
            client.connect(BROKER, MQTT_PORT)
            client.loop_start()
            mqtt_started = True
            print("✅ MQTT loop da khoi dong")
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