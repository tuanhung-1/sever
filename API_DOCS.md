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

Fall detection uses the trained v5 hybrid artifacts in:

- `artifacts/fall_detection/multistage/fall_v5_hybrid_deep.keras`
- `artifacts/fall_detection/multistage/fall_v5_metadata.json`
- `artifacts/fall_detection/multistage/fall_v5_sequence_mean.npy`
- `artifacts/fall_detection/multistage/fall_v5_sequence_std.npy`

Current model metadata:

- Window size: `320`
- Target rate: `100 Hz`
- Features: dual-input sequence + handcrafted features derived from the 8-channel IMU window
- Threshold: `0.73`
- Final alert confidence gate: `> 0.08`
- Post-filter: `post_fall_confirmation`

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
