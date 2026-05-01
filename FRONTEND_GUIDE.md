

**Sự kiện (Events)**:

1. **health_update** (50 Hz - Liên tục)
   - `data.bpm` (int: 0-220)
   - `data.spo2` (int: 70-100)
   - `data.temp` (double: 35-42°C)
   - `data.status` (String: NORMAL | HIGH_HEART_RATE | LOW_HEART_RATE | FEVER)
   - `fall.detected` (bool)
   - `fall.confidence` (double: 0-1)
   
   **Mẫu**:
   ```json
   {
     "data": {
       "bpm": 78,
       "spo2": 95,
       "temp": 36.5,
       "status": "NORMAL"
     },
     "fall": {
       "detected": false,
       "confidence": 0.0
     }
   }
   ```

2. **sensor_alert** (Cảnh báo ngã - Khi có va chạm)
   - `data.total_g` (double: Lực tác động)
   - `data.timestamp` (long)
   
   **Mẫu**:
   ```json
   {
     "data": {
       "total_g": 3.2,
       "timestamp": 1714576800000
     }
   }
   ```

3. **health_error** (Lỗi)
   - `message` (String)
   
   **Mẫu**:
   ```json
   {
     "message": "Invalid payload format"
   }
   ```

**API**:
- `GET /api/health/latest` - Dữ liệu mới nhất
- `GET /api/health/history?limit=50` - Lịch sử

