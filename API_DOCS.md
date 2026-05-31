# Backend API Docs (Frontend)

Base URL local:
- `http://127.0.0.1:5050`

## 1) API Docs

- **GET** `/api/docs`
- Muc dich: Lay tai lieu API dang JSON de frontend co the doc tu dong.

## 2) Latest Health

- **GET** `/api/health/latest`
- Query:
  - `format=packet` (mac dinh): goi health_update da duoc chuan hoa cho UI.
  - `format=raw`: payload raw da normalize tu ESP32.

### Response (format=packet)

```json
{
  "type": "health_update",
  "source_topic": "sensor/data",
  "timestamp": 1715000000,
  "data": {
    "bpm": 74,
    "spo2": 97.8,
    "temp": 36.2,
    "status": "NORMAL",
    "fall": {
      "detected": false,
      "confidence": 0.0
    }
  }
}
```

## 3) Health History

- **GET** `/api/health/history`
- Query:
  - `limit` (1..1000), mac dinh `50`.

### Response

```json
{
  "count": 2,
  "items": [
    {
      "type": "health_update",
      "source_topic": "sensor/data",
      "timestamp": 1715000000,
      "data": {
        "bpm": 74,
        "spo2": 97.8,
        "temp": 36.2,
        "status": "NORMAL"
      }
    }
  ]
}
```

## 4) Realtime (Socket.IO)

Frontend nen su dung Socket.IO de nhan realtime thay vi poll lien tuc.

Events:
- `health_update`: Ban tin suc khoe tong hop (co the kem ket qua fall AI).
- `sensor_alert`: Canh bao checking truoc khi xac nhan nga.
- `health_error`: Loi parse/validate payload.

## 5) Frontend Integration Notes

- Uu tien nguon realtime:
  - Lang nghe `health_update` de cap nhat UI.
- Fallback:
  - Khi vao app, goi `/api/health/latest?format=packet` de lay trang thai hien tai.
- Debug du lieu cam bien:
  - Dung `/api/health/latest?format=raw` de xem `ts`, `ts0`, `sample_interval_ms`, `ir/red`.
- BPM = 0:
  - Thuong do IR khong du quality (khong detect duoc peak), frontend nen hien thi trang thai "dang do" thay vi coi la nhip tim that.
