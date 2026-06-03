# Backend Server

Backend Flask + Socket.IO cho health monitor, MQTT sensor stream va fall detection.

## Structure

```text
app/
  main.py                  # Flask app, Socket.IO handlers, MQTT runtime
  core/
    config.py              # Environment settings
    extensions.py          # Shared Flask extensions
  models/
    health.py              # Health parsing and classification
    fall.py                # Fall detection model wrapper
  repositories/
    history_repository.py  # JSONL history persistence
artifacts/
  fall_detection/
    multistage/            # Trained CNN fall model artifacts
tests/                     # Smoke/regression tests
run.py                     # Local entrypoint
wsgi.py                    # WSGI entrypoint
```

Runtime history files are written to `storage/history/` by default. That folder is ignored by git and created automatically when needed.

The fall detector loads `artifacts/fall_detection/multistage/fall_cnn_gated_nolambda.keras` with its `fall_cnn_metadata.json`, `fall_cnn_mean.npy`, and `fall_cnn_std.npy` preprocessing artifacts.
The current trained model uses a 320-sample window, 8 motion features, threshold `0.87`, and the metadata `soft_peak_filter`.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
python run.py
```

API docs are available at `GET /api/docs`.
