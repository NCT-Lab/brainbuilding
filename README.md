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

#### Running the Service (Real-time Processing)

To start the real-time EEG service for live data processing from an LSL stream, use the `serve` command:

```bash
uv run python brainbuilding/service/cli.py serve --pipeline-config-path <path_to_pipeline_config> --state-config-path <path_to_state_config> --sfreq <sfreq>
```

#### Training

To train the pipeline components on recorded data, use the `train` command. Trained models are saved to the `models/` directory by default.

```bash
uv run python brainbuilding/service/cli.py train --sessions-root <path_to_data> --pipeline-config-path <path_to_pipeline_config> --state-config-path <path_to_state_config> --sfreq <sfreq>
```

#### Evaluation

To evaluate a trained pipeline on offline data, use the `evaluate` command:

```bash
uv run python brainbuilding/service/cli.py evaluate --sessions-root <path_to_data> --pipeline-config-path <path_to_pipeline_config> --state-config-path <path_to_state_config>
```

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



