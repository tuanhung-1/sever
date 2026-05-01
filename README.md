
## MQTT Topics

### 1. `sensor/data` (Vital Signs Batch)

Published every 1 second (50 samples @ 50 Hz)

```json
{
  "ts": 1710000000,
  "ts0": 1709999980,
  "temp": 36.5,
  "ir": [523456, 523500, 523520, ...],
  "red": [498123, 498200, 498240, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `ts` | int | Timestamp of last sample (Unix seconds) |
| `ts0` | int | Timestamp of first sample |
| `temp` | float | Body temperature (°C) |
| `ir` | [int] | IR LED readings (50 samples) → BPM |
| `red` | [int] | Red LED readings (50 samples) → SpO2 |

**Backend Processing**:
1. Extract IR/RED arrays
2. Peak detection on IR → BPM (35-220 range)
3. Ratio-based SpO2 from IR/RED
4. Classify: NORMAL (BPM 50-120), HIGH_HEART_RATE (>120), LOW_HEART_RATE (<50)
5. Emit `health_update` event

### 2. `sensor/alert` (Fall Impact)

Sent immediately on impact detection

```json
{
  "type": "fall_alert",
  "ts": 1710000000,
  "total_g": 2.75
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | str | Always `"fall_alert"` |
| `ts` | int | Impact timestamp |
| `total_g` | float | Acceleration magnitude (g-force) |

**Backend Processing**:
1. Detect fall_alert payload
2. Build alert packet
3. Emit `sensor_alert` event

### 3. `sensor/fall_raw` (200-Sample Window)

Sent when impact detected (4 seconds of motion data @ 50 Hz)

```json
{
  "event": "fall_raw",
  "trigger_ts": 1710000000,
  "ax": [0.12, 0.08, -0.21, ...],
  "ay": [-0.98, -1.03, -0.88, ...],
  "az": [9.81, 9.78, 9.74, ...],
  "gx": [0.01, 0.03, -0.02, ...],
  "gy": [0.02, 0.01, -0.01, ...],
  "gz": [0.00, -0.01, 0.02, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `event` | str | Always `"fall_raw"` |
| `trigger_ts` | int | Impact timestamp |
| `ax, ay, az` | [float] | Linear acceleration (m/s²), 200 samples |
| `gx, gy, gz` | [float] | Angular velocity (rad/s), 200 samples |

**Backend Processing**:
1. Extract 11 features (accel, gyro, roll, pitch, BPM, SpO2, temp)
2. Normalize with mean/std
3. Run through fall model (rule-based or AI LSTM)
4. Include result in `health_update` event

## REST API

### GET `/api/health/latest`

Get latest health packet

**Query**:
- `format=packet` (default) - Processed packet
- `format=raw` - Raw MQTT payload

**Response (packet)**:
```json
{
  "type": "health_update",
  "source_topic": "sensor/data",
  "server_timestamp": 1710000001,
  "data": {
    "ts": 1710000000,
    "bpm": 78,
    "spo2": 97,
    "temp": 36.5,
    "status": "NORMAL"
  },
  "fall": {
    "fall_detected": false,
    "confidence": 0.423,
    "confidence_percent": 42.3,
    "label": "NO_FALL",
    "state": "normal"
  }
}
```

**Response (raw)**:
```json
{
  "ts": 1710000000,
  "ts0": 1709999980,
  "temp": 36.5,
  "ir": [523456, 523500, 523520],
  "red": [498123, 498200, 498240],
  "ax": [], "ay": [], "az": [],
  "gx": [], "gy": [], "gz": []
}
```

### GET `/api/health/history`

Get saved health records

**Query**: `limit=50` (1-1000, default 50)

**Response**:
```json
{
  "count": 2,
  "items": [
    {
      "saved_at": 1710000001,
      "type": "health_update",
      "source_topic": "sensor/data",
      "server_timestamp": 1710000001,
      "data": {...},
      "fall": {...}
    }
  ]
}
```

## WebSocket Events

Connect: `ws://backend-ip:5050/socket.io`

### `health_update` (50 Hz)

```json
{
  "type": "health_update",
  "source_topic": "sensor/data",
  "server_timestamp": 1710000001,
  "data": {
    "ts": 1710000000,
    "bpm": 78,
    "spo2": 97,
    "temp": 36.5,
    "status": "NORMAL"
  },
  "fall": {
    "fall_detected": false,
    "confidence": 0.423,
    "confidence_percent": 42.3,
    "label": "NO_FALL",
    "state": "normal"
  }
}
```

**Status field**:
- `NORMAL` - Heart rate 50-120, temp < 38°C
- `HIGH_HEART_RATE` - BPM > 120
- `LOW_HEART_RATE` - BPM < 50
- `FEVER` - Temp > 38°C

### `sensor_alert`

```json
{
  "type": "fall_alert",
  "source_topic": "sensor/alert",
  "server_timestamp": 1710000001,
  "data": {
    "ts": 1710000000,
    "total_g": 2.75,
    "message": "fall_detected"
  }
}
```

### `health_error`

```json
{
  "type": "health_error",
  "source_topic": "sensor/data",
  "message": "Payload JSON khong hop le"
}
```

## Configuration

**Environment Variables** (app.py):
```bash
BROKER="dfee921e5f16440e8f3892ed3564c06d.s1.eu.hivemq.cloud"
MQTT_PORT=8883
USERNAME="hungdaica123"
PASSWORD="Hungpro123"
API_PORT=5050
HISTORY_FILE="health_history.jsonl"
USE_AI_FALL_MODEL="false"  # Use AI model or rule-based
MQTT_REQUIRED="false"      # Allow running without broker
```

**Thresholds** (model.py):
```python
BPM_HIGH_THRESHOLD = 120.0    # High heart rate
BPM_LOW_THRESHOLD = 50.0      # Low heart rate
TEMP_FEVER_THRESHOLD = 38.0   # Fever
```

**Fall Detection Thresholds** (fall_model.py):
```python
# Rule-based model
accel_threshold = 2.6  # g-force
gyro_threshold = 4.5   # rad/s

# AI model
AI_THRESHOLD = 0.6     # Probability
```

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask server, MQTT callbacks, WebSocket |
| `model.py` | Parse MQTT payload, estimate BPM/SpO2, classify status |
| `fall_model.py` | Fall detection (rule-based or AI LSTM) |
| `communication_summary.md` | Detailed protocol docs |
| `health_history.jsonl` | Saved health records (append-only) |
| `model.h5` | AI fall detection weights |
| `mean.npy`, `std.npy` | Normalization for AI model |

## Quick Test

```bash
# Parse sample payload
python3 << 'PY'
import json
from model import from_json_samples, classify

payload = {
    "ts": 1710000000,
    "ts0": 1709999980,
    "temp": 36.5,
    "ir": [523456] * 50,
    "red": [498123] * 50,
}

samples = from_json_samples(json.dumps(payload))
sample = samples[0]
status = classify(sample.heart_rate, sample.temp)
print(f"BPM={sample.heart_rate:.1f} SpO2={sample.spo2:.1f}% Status={status}")
PY

# Test API
curl http://localhost:5050/api/health/latest?format=packet
curl http://localhost:5050/api/health/history?limit=5
```

## WebSocket Test (JavaScript)

```javascript
const socket = io('http://localhost:5050');

socket.on('health_update', (data) => {
  console.log('BPM:', data.data.bpm, 'Status:', data.data.status);
});

socket.on('sensor_alert', (data) => {
  console.log('Fall alert!', data.data.total_g, 'g');
});

socket.on('health_error', (data) => {
  console.error('Error:', data.message);
});
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| MQTT connection failed | Broker unavailable | Check credentials, firewall (port 8883) |
| No BPM/SpO2 values | Missing IR/RED in payload | Verify sensor/data payload has `ir` and `red` arrays |
| Fall detection not working | Window not full | AI model needs 200 samples (4 sec). Check threshold. |
| WebSocket events not received | Connection issue | Check CORS, verify frontend connects to correct URL |

See `communication_summary.md` for full protocol details.
