## Brainbuilding Realtime EEG - Quickstart

### Prerequisites
- Python 3.11+
- uv
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Homebrew: `brew install uv`

### Install (using uv)
```bash
uv sync  # creates and populates .venv from pyproject.toml
```

You can run any command without activating the venv using `uv run`:
```bash
uv run python -V
```

### Train models (PT/CSP/SVC) from raw data
This builds the dataset from `data/raw-dataset-2` and saves artifacts to `models/`.
```bash
uv run python -m brainbuilding.service.cli train \
  --raw-data-dir data/raw-dataset-2 \
  --output-dir models
```
Artifacts produced:
- `models/pt.joblib`
- `models/csp.joblib`
- `models/classifier.joblib`

### Launch realtime service
Start the service with preloaded components and default TCP settings (127.0.0.1:12345):
```bash
uv run python -m brainbuilding.service.cli \
  --preload-dir models
```
Notes:
- Expects LSL streams named `NeoRec21-1247` (EEG) and `Brainbuilding-Events` (events).
- Sends results over TCP to 127.0.0.1:12345 by default.

### Mock TCP server (receiver)
Listen for realtime results locally:
```bash
uv run python scripts/mocks/tcp_server.py
```

### Mock LSL endpoint (replay XDF + events)
Replays the XDF and emits event codes from the `Task.json` timeline:
```bash
uv run python scripts/mocks/mock_lsl_endpoint.py
```

Recommended startup order for local testing:
1) Mock TCP server
2) Mock LSL endpoint
3) Realtime service

### Event codes
The service consumes numeric event codes via LSL:
- 1: EyeWarmupDesc
- 2: EyeWarmupBlink
- 3: EyeWarmupText
- 4: EyeWarmupMove
- 5: Background
- 10: Task period (Rest/Attention/Image/Point/Vas)


