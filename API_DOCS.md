# Backend API Docs

Base URL local:

- `http://127.0.0.1:5050`

## HTTP

- `GET /api/docs`
  - Returns machine-readable API and Socket.IO documentation.

- `GET /api/history`
  - Query:
    - `limit`: `1..1000`, default `50`
    - `type`: `all | health | fall`, default `all`
  - Returns saved health and fall update history from `storage/history/`.

- `GET /api/training`
  - Returns counts for collected fall raw training windows.

- `GET /api/training/fall.csv`
  - Downloads flattened CSV for fall training data collected after the ESP32 `.ino` filter.

## AI Model

Fall detection uses the trained multi-stage gated 1D CNN artifacts in:

- `artifacts/fall_detection/multistage/fall_cnn_gated_nolambda.keras`
- `artifacts/fall_detection/multistage/fall_cnn_metadata.json`
- `artifacts/fall_detection/multistage/fall_cnn_mean.npy`
- `artifacts/fall_detection/multistage/fall_cnn_std.npy`

Current model metadata:

- Window size: `320`
- Target rate: `100 Hz`
- Features: `ax, ay, az, gx, gy, gz, acc_mag, gyro_mag`
- Threshold: `0.87`
- Post-filter: `soft_peak_filter`

## Socket.IO

Frontend should subscribe to:

- `health_update`
- `fall_update`
- `health_error`
- `buzz_response`

Frontend can emit:

- `buzz`
  - Payload: `{ "action": "off", "reason": "user_dismissed", "timestamp": 1715000000 }`

## Run

```powershell
python run.py
```
