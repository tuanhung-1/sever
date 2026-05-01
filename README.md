
# Backend API - Frontend Guide

## WebSocket Events

Connect: `ws://backend-ip:5050/socket.io`

### `health_update` (50 Hz - Liên tục)

Vital signs + fall detection data

```json
{
  "data": {
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

### `sensor_alert` (Cảnh báo ngã)

```json
{
  "data": {
    "total_g": 2.75
  }
}
```

### `health_error` (Lỗi)

```json
{
  "message": "Invalid payload"
}
```

## REST API

### GET `/api/health/history?limit=50`

Get saved health records (limit 1-1000)

**Response**:
```json
{
  "items": [
    {
      "data": {
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
  ]
}
```


