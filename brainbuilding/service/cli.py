"""
Command-line interface for the brainbuilding real-time EEG processor.

Refactored to a strictly-typed, declarative Typer app with explicit
subcommand handlers and Pydantic option models.
"""

import logging
import os
from typing import List, Optional

import numpy as np
import typer  # type: ignore
from joblib import dump as joblib_dump  # type: ignore
from pydantic import BaseModel, Field

from brainbuilding.core.config import (
    CHANNELS_IN_STREAM,
    CHANNELS_TO_KEEP,
    DEFAULT_EEG_STREAM_NAME,
    DEFAULT_EVENT_STREAM_NAME,
    DEFAULT_HISTORY_PATH,
    DEFAULT_PROCESSING_STEP,
    DEFAULT_PROCESSING_WINDOW_SIZE,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_SFREQ,
    DEFAULT_TCP_HOST,
    DEFAULT_TCP_PORT,
    DEFAULT_TCP_RETRIES,
    REFERENCE_CHANNEL,
)
from brainbuilding.service.pipeline import load_pipeline_from_yaml
from brainbuilding.service.eeg_service import StateCheckRunner, EEGEvaluationRunner
from sklearn.metrics import f1_score, roc_auc_score


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)04d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s.%(msecs)04d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


class ServeOptions(BaseModel):
    # Stream configuration
    eeg_stream_name: str = DEFAULT_EEG_STREAM_NAME
    event_stream_name: str = DEFAULT_EVENT_STREAM_NAME

    # Channel configuration
    channels: List[str] = Field(
        default_factory=lambda: list(CHANNELS_IN_STREAM)
    )
    channels_to_keep: List[str] = Field(
        default_factory=lambda: list(CHANNELS_TO_KEEP)
    )
    reference_channel: str = REFERENCE_CHANNEL

    # Processing parameters
    processing_window_size: int = DEFAULT_PROCESSING_WINDOW_SIZE
    processing_step: int = DEFAULT_PROCESSING_STEP
    sfreq: float = DEFAULT_SFREQ
    session_id: Optional[str] = None
    scale_factor: float = DEFAULT_SCALE_FACTOR

    # TCP configuration
    tcp_host: str = DEFAULT_TCP_HOST
    tcp_port: int = DEFAULT_TCP_PORT
    tcp_retries: int = DEFAULT_TCP_RETRIES

    # File paths
    pipeline_config: str = "configs/pipeline_config.yaml"
    history_path: str = DEFAULT_HISTORY_PATH
    state_config: Optional[str] = "configs/states/state_config.yaml"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    debug: bool = False

    # Preloading
    preload_dir: str = "models"
    no_preload: bool = False


app = typer.Typer(help="Real-time EEG processing for brain-computer interfaces")


@app.command("serve")
def serve(
    channels: List[str] = typer.Option(list(CHANNELS_IN_STREAM), help="Channels in stream"),
    channels_to_keep: List[str] = typer.Option(list(CHANNELS_TO_KEEP), help="Channels to keep for processing"),
    reference_channel: str = typer.Option(REFERENCE_CHANNEL, help="Reference channel name"),
    sfreq: float = typer.Option(DEFAULT_SFREQ, help="Sampling frequency (Hz)"),
    session_id: Optional[str] = typer.Option(None, help="Session identifier"),
    scale_factor: float = typer.Option(DEFAULT_SCALE_FACTOR, help="Scale factor for unit conversion"),
    tcp_host: str = typer.Option(DEFAULT_TCP_HOST, help="TCP host for results"),
    tcp_port: int = typer.Option(DEFAULT_TCP_PORT, help="TCP port"),
    tcp_retries: int = typer.Option(DEFAULT_TCP_RETRIES, help="Max TCP retries"),
    pipeline_config_path: str = typer.Option("configs/pipeline_config.yaml", help="Pipeline YAML path"),
    state_config_path: Optional[str] = typer.Option("configs/states/state_config.yaml", help="State machine YAML path"),
    preload_dir: str = typer.Option("models", help="Preload directory for components"),
    no_preload: bool = typer.Option(False, help="Disable preloading components"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, help="Log file path"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    setup_logging(log_level, log_file)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    from brainbuilding.service.eeg_service import EEGService

    pipeline_config, pretrained = load_pipeline_from_yaml(
        pipeline_config_path,
        preload_dir=(None if no_preload else preload_dir),
        enable_preload=not no_preload,
    )

    service = EEGService(
        pipeline_config,
        tcp_host=tcp_host,
        tcp_port=tcp_port,
        tcp_retries=tcp_retries,
        state_config_path=state_config_path,
        session_id=session_id,
        sfreq=sfreq,
    )

    if pretrained:
        service.state_manager.fitted_components.update(pretrained)
        service.state_manager.pipeline_ready = (
            service.state_manager._have_all_components()
        )
        keys = list(pretrained.keys())
        ready = service.state_manager.pipeline_ready
        logging.info(
            "Preloaded components: keys_count=%d ready=%s", len(keys), ready
        )

    try:
        service.run()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()


@app.command("train")
def train(
    calibrated_array_path: str = typer.Option("data/calibrated.npy", help="Path to save/load calibrated array"),
    save_calibrated: bool = typer.Option(True, help="Save calibrated array to disk"),
    use_calibrated: bool = typer.Option(False, help="Use pre-saved calibrated array"),
    sessions_root: str = typer.Option("data", help="Root directory containing session subfolders"),
    sessions_dirs: Optional[List[str]] = typer.Option(None, help="Explicit list of session directories"),
    output_dir: str = typer.Option("models", help="Output directory for trained components"),
    exclude_label: int = typer.Option(2, help="Exclude samples with this label (-1 to disable)"),
    pipeline_config_path: str = typer.Option("configs/pipeline_config.yaml", help="Pipeline YAML path for training"),
    state_config_path: str = typer.Option("configs/states/state_config.yaml", help="State machine YAML path"),
    preload_dir: str = typer.Option("models", help="Preload directory for components"),
    no_preload: bool = typer.Option(False, help="Disable preloading components"),
    sfreq: float = typer.Option(DEFAULT_SFREQ, help="Sampling frequency (Hz)"),
) -> None:
    setup_logging("INFO", None)

    pipeline_config, _ = load_pipeline_from_yaml(
        pipeline_config_path,
        preload_dir=(None if no_preload else preload_dir),
        enable_preload=not no_preload,
    )

    if not use_calibrated:
        from brainbuilding.service.eeg_service import EEGOfflineRunner

        if sessions_dirs is not None and len(sessions_dirs) > 0:
            session_dirs = [d for d in sessions_dirs if os.path.isdir(d)]
        else:
            session_dirs = [
                os.path.join(sessions_root, d)
                for d in os.listdir(sessions_root)
                if os.path.isdir(os.path.join(sessions_root, d))
            ]

        all_rows: List[np.ndarray] = []
        for sess_dir in sorted(session_dirs):
            sess_id = os.path.basename(sess_dir)
            xdf_path = os.path.join(sess_dir, "data.xdf")
            if not os.path.exists(xdf_path):
                logging.info("Skipping %s because it does not exist", xdf_path)
                continue
            logging.info("Loading %s", xdf_path)
            runner = EEGOfflineRunner(
                pipeline_config=pipeline_config,
                state_config_path=state_config_path,
                sfreq=sfreq,
            )
            rows = runner.run_from_xdf(xdf_path, session_id=sess_id)
            all_rows.extend(rows)

        data = np.concatenate(all_rows) if len(all_rows) > 0 else np.array([])
        if data.size == 0:
            raise RuntimeError("No calibrated data produced from provided sessions")

        if save_calibrated:
            np.save(calibrated_array_path, data)
        all_rows_arr = data
    else:
        all_rows_arr = np.load(calibrated_array_path)

    logging.info("%s", f"{all_rows_arr.shape=}")
    logging.info("%s", f"{all_rows_arr['sample'].shape=}")

    y = all_rows_arr["label"]
    training_cfg = pipeline_config.training_only_config()
    fitted = training_cfg.fit_training(all_rows_arr, y)

    os.makedirs(output_dir, exist_ok=True)
    for name, comp in fitted.items():
        outp = os.path.join(output_dir, f"{name}.joblib")
        joblib_dump(comp, outp)
    logging.info("Saved %d components to %s", len(fitted), output_dir)


@app.command("check-states")
def check_states(
    sessions_root: str = typer.Option(
        "data", help="Root directory containing session subfolders"
    ),
    sessions_dirs: Optional[List[str]] = typer.Option(
        None, help="Explicit list of session directories"
    ),
    state_config_path: str = typer.Option(
        "configs/states/state_config.yaml", help="State machine YAML path"
    ),
) -> None:
    """Validate state machine transitions against task definitions."""

    setup_logging("INFO", None)

    if sessions_dirs is not None and len(sessions_dirs) > 0:
        session_dirs = [d for d in sessions_dirs if os.path.isdir(d)]
    else:
        session_dirs = [
            os.path.join(sessions_root, d)
            for d in os.listdir(sessions_root)
            if os.path.isdir(os.path.join(sessions_root, d))
        ]

    runner = StateCheckRunner(state_config_path=state_config_path)
    total_mismatches = 0

    for sess_dir in sorted(session_dirs):
        xdf_path = os.path.join(sess_dir, "data.xdf")
        task_json_path = os.path.join(sess_dir, "task.json")

        if not os.path.exists(xdf_path) or not os.path.exists(task_json_path):
            logging.warning(
                "Skipping %s: missing data.xdf or task.json", sess_dir
            )
            continue

        logging.info("Checking states for session: %s", sess_dir)
        mismatches = runner.run_check(xdf_path, task_json_path)

        if mismatches:
            logging.error(
                "Found %d mismatches in %s:", len(mismatches), sess_dir
            )
            for i, (expected, actual) in mismatches.items():
                logging.error(
                    "  Event index %d: Expected '%s', got '%s'",
                    i,
                    expected,
                    actual,
                )
            total_mismatches += len(mismatches)
        else:
            logging.info("Session %s OK: All states matched.", sess_dir)

    if total_mismatches > 0:
        logging.error("State check failed with %d total mismatches.", total_mismatches)
    else:
        logging.info("State check passed for all sessions.")


@app.command("evaluate")
def evaluate(
    sessions_root: str = typer.Option(
        "data", help="Root directory containing session subfolders"
    ),
    sessions_dirs: Optional[List[str]] = typer.Option(
        None, help="Explicit list of session directories"
    ),
    pipeline_config_path: str = typer.Option(
        "configs/pipeline/pipeline_config.yaml", help="Pipeline YAML path for evaluation"
    ),
    state_config_path: str = typer.Option(
        "configs/states/state_config.yaml", help="State machine YAML path"
    ),
    preload_dir: str = typer.Option("models", help="Preload directory for components"),
    no_preload: bool = typer.Option(False, help="Disable preloading components"),
    sfreq: float = typer.Option(DEFAULT_SFREQ, help="Sampling frequency (Hz)"),
) -> None:
    """Evaluate pipeline performance against offline data."""
    setup_logging("INFO", None)

    pipeline_config, pretrained = load_pipeline_from_yaml(
        pipeline_config_path,
        preload_dir=(None if no_preload else preload_dir),
        enable_preload=not no_preload,
    )

    if sessions_dirs is not None and len(sessions_dirs) > 0:
        session_dirs = [d for d in sessions_dirs if os.path.isdir(d)]
    else:
        session_dirs = [
            os.path.join(sessions_root, d)
            for d in os.listdir(sessions_root)
            if os.path.isdir(os.path.join(sessions_root, d))
        ]

    runner = EEGEvaluationRunner(
        pipeline_config=pipeline_config,
        state_config_path=state_config_path,
        sfreq=sfreq,
        pretrained_components=pretrained,
    )
    runner.pipeline_config.steps = [
        s for s in runner.pipeline_config.steps if not s.is_fit_during_runtime
    ]

    all_predictions_agg = []
    all_ground_truth_agg = []
    all_probabilities_agg = []
    all_class_probabilities_agg = []

    for sess_dir in sorted(session_dirs):
        sess_id = os.path.basename(sess_dir)
        xdf_path = os.path.join(sess_dir, "data.xdf")
        if not os.path.exists(xdf_path):
            logging.info("Skipping %s because it does not exist", xdf_path)
            continue
        logging.info("Evaluating session: %s", sess_dir)

        (
            predictions,
            ground_truth,
            probabilities,
            class_probabilities,
        ) = runner.run_from_xdf(xdf_path, session_id=sess_id)

        if not ground_truth:
            logging.warning("No data to evaluate for session %s.", sess_dir)
            continue

        # Per-session stats
        f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
        logging.info(f"  Session F1 Score (weighted): {f1:.4f}")

        if class_probabilities:
            try:
                y_true_sess = np.array(ground_truth)
                y_score_sess = np.array(class_probabilities)

                if y_score_sess.ndim == 2 and y_score_sess.shape[1] == 2:
                    auc = roc_auc_score(y_true_sess, y_score_sess[:, 1])
                    logging.info(f"  Session AUC Score: {auc:.4f}")
                else:
                    auc = roc_auc_score(
                        y_true_sess,
                        y_score_sess,
                        multi_class="ovr",
                        average="weighted",
                    )
                    logging.info(f"  Session AUC Score (weighted OVR): {auc:.4f}")
            except ValueError as e:
                logging.warning(f"  Could not calculate session AUC score: {e}")

        # Aggregate results
        all_predictions_agg.extend(predictions)
        all_ground_truth_agg.extend(ground_truth)
        all_probabilities_agg.extend(probabilities)
        all_class_probabilities_agg.extend(class_probabilities)

    if not all_ground_truth_agg:
        logging.warning("No data to evaluate across all sessions.")
        return

    # Final aggregate stats
    logging.info("=" * 20)
    logging.info("Aggregate Evaluation Results")
    logging.info("=" * 20)

    final_f1 = f1_score(
        all_ground_truth_agg, all_predictions_agg, average="weighted", zero_division=0
    )
    logging.info(f"Overall F1 Score (weighted): {final_f1:.4f}")

    if all_class_probabilities_agg:
        try:
            y_true_agg = np.array(all_ground_truth_agg)
            y_score_agg = np.array(all_class_probabilities_agg)
            if y_score_agg.ndim == 2 and y_score_agg.shape[1] == 2:
                final_auc = roc_auc_score(y_true_agg, y_score_agg[:, 1])
                logging.info(f"Overall AUC Score: {final_auc:.4f}")
            else:
                final_auc = roc_auc_score(
                    y_true_agg,
                    y_score_agg,
                    multi_class="ovr",
                    average="weighted",
                )
                logging.info(f"Overall AUC Score (weighted OVR): {final_auc:.4f}")
        except ValueError as e:
            logging.warning(f"Could not calculate overall AUC score: {e}")


def main(argv: Optional[List[str]] = None) -> None:
    app()


if __name__ == "__main__":
    main()
