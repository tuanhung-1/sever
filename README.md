
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
    "detected": false,
    "confidence": 0.423
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
    "detected": false,
    "confidence": 0.423
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


**Thresholds** (model.py):
```python
BPM_HIGH_THRESHOLD = 120.0    # High heart rate
BPM_LOW_THRESHOLD = 50.0      # Low heart rate
TEMP_FEVER_THRESHOLD = 38.0   # Fever
```

