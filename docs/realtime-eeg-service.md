## Realtime EEG Service – Updated Architecture and Runtime Behavior

### Overview
- Realtime process that reads EEG and event LSL streams, performs online preprocessing and inference, and streams results over TCP.
- Strict pipeline order with minimal coordination logic. Stateless steps instantiate on demand; stateful steps are fitted incrementally.

### Data flow (high-level)
1) LSL inlets
   - EEG: `StreamInlet` for samples
   - Events: `StreamInlet` for state transitions
2) Online preprocessing per sample (`PointProcessor`)
   - Reference subtraction (Fz)
   - Scaling µV→V
   - Channel mask
   - Stateful bandpass filter
3) Sliding window
   - `window_size` (default 250 pts), step `step_size` (default 125 pts)
4) Pipeline execution (async in ProcessPool)
   - See Pipeline section for fit/partial_fit/inference behavior
5) TCP sender
   - Handshake message on connect
   - Newline-delimited JSON messages for results

### State machine (events → behavior)
- States: `IDLE`, `EYE_WARMUP_DESC`, `EYE_WARMUP_BLINK`, `EYE_WARMUP_TEXT`, `EYE_WARMUP_MOVE`, `BACKGROUND`, `INFERENCE`.
- Data collection groups:
  - Eye movement states → `EYE_MOVEMENT` group
  - `BACKGROUND` → `BACKGROUND` group
  - `INFERENCE` → `INFERENCE` group
- Transition-specific actions:
  - `EYE_WARMUP_MOVE → BACKGROUND` triggers FIT
  - `INFERENCE` entry enables real-time prediction and periodic partial_fit

### Pipeline system
- Defined in `PipelineConfig.steps` with `PipelineStep` entries:
  - `component_class`, `init_params`, `requires_fit`, `is_classifier`
- Fitting logic
  - Stateless steps (requires_fit=False) are instantiated as needed
  - Stateful steps are fitted one-by-one in order via `fit_next_stateful(data, fitted)`
  - FIT is triggered by state transitions (see above)
- Partial fit
  - In `INFERENCE` state, periodic partial-fits submit non-overlapping windows from `INFERENCE` group
  - Only steps that implement partial_fit are updated

### Inference
- No gating on “pipeline_ready”: inference runs as soon as a full window exists and `INFERENCE` is active.
- Transform chain:
  - Runs steps strictly in order on the window in a process pool job
  - Errors do not crash the service:
    - Missing component / invalid data are logged and the inference for that window is skipped (returns `null` to the callback)
- Classifier output
  - Classifier wrapper attempts `predict_proba` and returns `[pred, proba]`
  - Fallback: `predict` with `proba=1.0`

### TCP sender
- On connect, sends handshake:
  - `{ "type": "handshake", "client_name": "eeg-processor" }\n`
- For each result, sends a newline-delimited JSON object:
  - If the payload is a dict without `type`, service adds `"type": "data"`
  - Typical data message:
    - `{ "type": "data", "prediction": <int>, "probability": <float>, "pipeline_type": "hybrid", "processing_time": <float>, "window_metadata": { ... } }\n`
- Newline (`\n`) is the message delimiter; server should parse stream line-by-line

### CLI entry (serve mode)
```
uv run python -m brainbuilding.service.cli \
  --preload-dir models \
  --tcp-host localhost \
  --tcp-port 8080
```
- Optional: logging flags `--log-level`, `--log-file`
- TCP host/port/retries configurable via CLI; passed to `TCPSender`

### Preloading models
- On startup, components may be preloaded (e.g., PT/CSP/SVC) from the `models` dir
- Preloaded items are placed into `state_manager.fitted_components`
  - CSP/SVC are considered ready to use
  - PT will still need partial_fit data to reach its internal sample threshold before it can produce stable outputs

### Extending the pipeline
- Add a step in `PipelineConfig.hybrid_pipeline()` with:
  - `name`, `component_class`, `init_params`, `requires_fit`, `is_classifier`
- Design rules:
  - Stateless steps should transform input without fitting and be marked `requires_fit=False`
  - Stateful steps should implement `fit` and (optionally) `partial_fit`
  - Classifiers should expose `predict_proba` when probabilities are needed

### Mock TCP server (for testing)
- The mock server in `scripts/mocks/tcp_server.py` reads newline-delimited JSON and logs each line immediately upon receipt.
- Useful for validating handshake and data message streams in real time.

### Known behavior and caveats
- PT (Parallel Transport) may be “fit” but still reject inference until enough subject samples accumulate via `partial_fit`. This is expected; inference logs the failure and continues.
- Inference may be skipped until required steps become available; the service continues running and will start producing predictions as soon as the pipeline can process windows end-to-end.


