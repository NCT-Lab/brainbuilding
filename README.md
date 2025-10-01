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


### Usage

All commands should be run from the project root.

**A Note on File Paths**

When you provide a path to a configuration file or a model directory (e.g., `configs/pipeline/pipeline_config.yaml`), the application searches for it in two places, in order:

1.  **Relative to your current directory:** It first checks if the path exists where you are running the command.
2.  **Inside the installed package:** If the path is not found in your current directory, it looks inside the installed `brainbuilding` package.

#### Running the Service (Real-time Processing)

To start the real-time EEG service for live data processing from an LSL stream, use the `serve` command:

```bash
# Serve with default configs (models will be preloaded from 'models/')
uv run python -m brainbuilding.service.cli serve --pipeline-config-path configs/pipeline/pipeline_config.yaml --state-config-path configs/states/state_config_new.yaml
```

#### Training

To train the pipeline components on recorded data, use the `train` command. By default, trained models are saved to the `brainbuilding/models/` directory inside the package.

```bash
# Train on data in `data/` using default configs, save models to `models/`
uv run python -m brainbuilding.service.cli train --sessions-root data/ --pipeline-config-path configs/pipeline/pipeline_config.yaml --state-config-path configs/states/state_config_old_train.yaml

# Train and save models to a different directory
uv run python -m brainbuilding.service.cli train --sessions-root data/ --output-dir my-trained-models/ --pipeline-config-path configs/pipeline/pipeline_config.yaml --state-config-path configs/states/state_config_old_train.yaml
```

#### Evaluation

To evaluate a trained pipeline on offline data, use the `evaluate` command:

```bash
# Evaluate with default configs and preloaded models from 'models/'
uv run python -m brainbuilding.service.cli evaluate --sessions-root data/ --pipeline-config-path configs/pipeline/pipeline_config.yaml --state-config-path configs/states/state_config_new.yaml
```

### Command Reference

All commands support the `--help` flag for a full list of options (e.g., `uv run python -m brainbuilding.service.cli serve --help`).

#### `serve`

Starts the real-time EEG service to process a live LSL stream.

**Arguments:**
-   `--channels-to-keep`: Channels to use for processing (default: predefined list).
-   `--reference-channel`: Name of the reference channel (default: `CPz`).
-   `--session-id`: A unique identifier for the session, which must be provided manually. This is critical for pipeline steps that track state across sessions (e.g., `ParallelTransport`).
-   `--tcp-host`: Host for the TCP results server (default: `127.0.0.1`).
-   `--tcp-port`: Port for the TCP server (default: `12345`).
-   `--tcp-retries`: Max TCP connection retries (default: `3`).
-   `--pipeline-config-path`: Path to pipeline config.
-   `--state-config-path`: Path to state machine config.
-   `--preload-dir`: Directory to preload models from (default: `models`).
-   `--no-preload`: Disable model preloading (optional).
-   `--log-level`: Logging level (e.g., `INFO`, `DEBUG`).
-   `--log-file`: Path to an optional log file.
-   `--debug`: Enable debug-level logging.

#### `train`

Trains pipeline components using offline data and saves them as `.joblib` artifacts.

**Arguments:**
-   `--calibrated-array-path`: Path to load/save calibrated data (`.npy`).
-   `--save-calibrated`: Save the calibrated data array.
-   `--use-calibrated`: Use a pre-saved calibrated data array instead of processing from scratch.
-   `--evaluate-losocv`: Perform Leave-One-Session-Out Cross-Validation during training.
-   `--sessions-root`: Root directory containing session data folders (default: `data`).
-   `--sessions-dirs`: Explicit list of session directories to use. The name of each directory is used as the `session-id`.
-   `--output-dir`: Directory to save trained models (default: `models`).
-   `--exclude-label`: Label to exclude from training data (default: `2`).
-   `--pipeline-config-path`: Path to pipeline config.
-   `--state-config-path`: Path to state machine config.
-   `--preload-dir`: Directory to preload models from (default: `models`).
-   `--no-preload`: Disable model preloading (by default is passed for training).

#### `evaluate`

Evaluates a trained pipeline on offline session data.

**Arguments:**
-   `--sessions-root`: Root directory containing session data folders (default: `data`).
-   `--sessions-dirs`: Explicit list of session directories to evaluate.
-   `--pipeline-config-path`: Path to pipeline config.
-   `--state-config-path`: Path to state machine config.
-   `--preload-dir`: Directory to preload models from (default: `models`).
-   `--no-preload`: Disable model preloading.
-   `--channels-to-keep`: Channels to use for processing.
-   `--reference-channel`: Name of the reference channel.

#### `check-states`

Validates state machine transitions against recorded LSL markers in offline data.

**Arguments:**
-   `--sessions-root`: Root directory containing session data folders (default: `data`).
-   `--sessions-dirs`: Explicit list of session directories to check.
-   `--state-config-path`: Path to the state machine config to validate.

### Configuration Files

Service behavior is defined by YAML configuration files in the `configs/` directory.

#### Pipeline Configs (`configs/pipeline/`)

These files define the signal processing steps.

-   `pipeline_config.yaml`: For data sampled at 250Hz.
-   `pipeline_config_downsample.yaml`: For data sampled at 500Hz (includes a downsampling step).

#### State Machine Configs (`configs/states/`)

These files control when and how data is processed based on experimental events.

-   **Old dataset:**
    -   `state_config_old_train.yaml`: Use for training.
    -   `state_config_old.yaml`: Use for evaluation.
-   **New dataset:**
    -   `state_config_new.yaml`: Runs inference only during core rest and motor imagery phases.
    -   `state_config_new_inference_all.yaml`: Runs inference during all motor imagery-related phases, including intermediate focus and feedback periods.



