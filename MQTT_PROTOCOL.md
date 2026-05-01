# MQTT Topics & Data Samples

Broker: `dfee921e5f16440e8f3892ed3564c06d.s1.eu.hivemq.cloud:8883`

---

## 1. `sensor/data` (Vital Signs - Every 1s)

```json
{
  "ts": 1714576800,
  "ts0": 1714576780,
  "temp": 36.5,
  "ir": [523456, 523500, 523520, ...],
  "red": [498123, 498200, 498240, ...]
}
```

---

## 2. `sensor/alert` (Fall Impact - Immediate)

```json
{
  "type": "fall_alert",
  "ts": 1714576800,
  "total_g": 2.75
}
```

---

## 3. `sensor/fall_raw` (200 Motion Samples - On Impact)

```json
{
  "event": "fall_raw",
  "trigger_ts": 1714576800,
  "ax": [0.12, 0.08, -0.21, ...],
  "ay": [-0.98, -1.03, -0.88, ...],
  "az": [9.81, 9.78, 9.74, ...],
  "gx": [0.01, 0.03, -0.02, ...],
  "gy": [0.02, 0.01, -0.01, ...],
  "gz": [0.00, -0.01, 0.02, ...]
}
```

---

## 4. `device/command` (Backend → Device)

```json
{
  "cmd": "buzzer_on",
  "duration_ms": 5000
}
```
