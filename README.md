
# Backend API - Socket.IO Guide

### Flutter/Dart

```dart
import 'package:socket_io_client/socket_io_client.dart' as IO;

void main() {
  final socket = IO.io('http://backend-ip:5050', IO.OptionBuilder()
    .setTransports(['websocket'])
    .disableAutoConnect()
    .build());

  socket.connect();

  // Real-time health data (50 Hz)
  socket.on('health_update', (data) {
    final bpm = data['data']['bpm'];
    final spo2 = data['data']['spo2'];
    final temp = data['data']['temp'];
    final status = data['data']['status'];
    final fallDetected = data['fall']['detected'];
    
    print('❤️ $bpm bpm | O₂ $spo2% | 🌡️ $temp°C | Status: $status');
    if (fallDetected) {
      print('🚨 FALL DETECTED!');
    }
  });

  // Fall impact alert
  socket.on('sensor_alert', (data) {
    final totalG = data['data']['total_g'];
    print('⚠️ Impact: ${totalG}g');
  });

  // Errors
  socket.on('health_error', (data) {
    print('❌ Error: ${data['message']}');
  });

  // Connection events
  socket.onConnect((_) => print('✅ Connected'));
  socket.onDisconnect((_) => print('❌ Disconnected'));
  socket.onError((error) => print('Error: $error'));
}
```

## WebSocket Connection

Connect to: `ws://backend-ip:5050/socket.io`

## Events

### `health_update` (50 Hz - Liên tục)

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

### `sensor_alert` (Cảnh báo ngã - Khi có va chạm)

```json
{
  "data": {
    "total_g": 2.75
  }
}
```

### `health_error` (Lỗi parse)

```json
{
  "message": "Invalid payload"
}
```

## REST API (Alternative)

### GET `/api/health/latest`

Lấy dữ liệu mới nhất

**JavaScript/React**:
```javascript
fetch('http://backend-ip:5050/api/health/latest?format=packet')
  .then(res => res.json())
  .then(data => {
    console.log(data.data.bpm);      // Nhịp tim
    console.log(data.fall.detected); // Ngã?
  });
```

**Flutter/Dart**:
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

final response = await http.get(
  Uri.parse('http://backend-ip:5050/api/health/latest?format=packet')
);
final data = jsonDecode(response.body);
print('BPM: ${data['data']['bpm']}');
```

**Response**:
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

### GET `/api/health/history?limit=50`


```

**Response**:
```json
{
  "count": 5,
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


