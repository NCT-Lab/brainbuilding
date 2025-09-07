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
    EEGProcessingConfig,
)
from brainbuilding.service.pipeline import load_pipeline_from_yaml


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


class TrainOptions(BaseModel):
    calibrated_array_path: str = "data/calibrated.npy"
    save_calibrated: bool = True
    use_calibrated: bool = False

    sessions_root: str = "data"
    sessions_dirs: Optional[List[str]] = None
    output_dir: str = "models"
    exclude_label: int = 2

    pipeline_config: str = "configs/pipeline_config.yaml"
    state_config: str = "configs/states/state_config.yaml"

    preload_dir: str = "models"
    no_preload: bool = False


def options_to_processing_config(opts: ServeOptions) -> EEGProcessingConfig:
    return EEGProcessingConfig(
        channels=opts.channels,
        reference_channel=opts.reference_channel,
        processing_window_size=opts.processing_window_size,
        processing_step=opts.processing_step,
        sfreq=opts.sfreq,
        eeg_stream_name=opts.eeg_stream_name,
        event_stream_name=opts.event_stream_name,
        tcp_host=opts.tcp_host,
        tcp_port=opts.tcp_port,
        tcp_retries=opts.tcp_retries,
        pipeline_path=opts.pipeline_config,
        history_path=opts.history_path,
        scale_factor=opts.scale_factor,
        debug=opts.debug,
    )


app = typer.Typer(help="Real-time EEG processing for brain-computer interfaces")


@app.command("serve")
def serve(
    eeg_stream_name: str = typer.Option(DEFAULT_EEG_STREAM_NAME, help="EEG LSL stream name"),
    event_stream_name: str = typer.Option(DEFAULT_EVENT_STREAM_NAME, help="Event LSL stream name"),
    channels: List[str] = typer.Option(list(CHANNELS_IN_STREAM), help="Channels in stream"),
    channels_to_keep: List[str] = typer.Option(list(CHANNELS_TO_KEEP), help="Channels to keep for processing"),
    reference_channel: str = typer.Option(REFERENCE_CHANNEL, help="Reference channel name"),
    processing_window_size: int = typer.Option(DEFAULT_PROCESSING_WINDOW_SIZE, help="Window size (samples)"),
    processing_step: int = typer.Option(DEFAULT_PROCESSING_STEP, help="Step size (samples)"),
    sfreq: float = typer.Option(DEFAULT_SFREQ, help="Sampling frequency (Hz)"),
    session_id: Optional[str] = typer.Option(None, help="Session identifier"),
    scale_factor: float = typer.Option(DEFAULT_SCALE_FACTOR, help="Scale factor for unit conversion"),
    tcp_host: str = typer.Option(DEFAULT_TCP_HOST, help="TCP host for results"),
    tcp_port: int = typer.Option(DEFAULT_TCP_PORT, help="TCP port"),
    tcp_retries: int = typer.Option(DEFAULT_TCP_RETRIES, help="Max TCP retries"),
    pipeline_config_path: str = typer.Option("configs/pipeline_config.yaml", help="Pipeline YAML path"),
    history_path: str = typer.Option(DEFAULT_HISTORY_PATH, help="History output path"),
    state_config_path: Optional[str] = typer.Option("configs/states/state_config.yaml", help="State machine YAML path"),
    preload_dir: str = typer.Option("models", help="Preload directory for components"),
    no_preload: bool = typer.Option(False, help="Disable preloading components"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, help="Log file path"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
) -> None:
    opts = ServeOptions(
        eeg_stream_name=eeg_stream_name,
        event_stream_name=event_stream_name,
        channels=channels,
        channels_to_keep=channels_to_keep,
        reference_channel=reference_channel,
        processing_window_size=processing_window_size,
        processing_step=processing_step,
        sfreq=sfreq,
        session_id=session_id,
        scale_factor=scale_factor,
        tcp_host=tcp_host,
        tcp_port=tcp_port,
        tcp_retries=tcp_retries,
        pipeline_config=pipeline_config_path,
        history_path=history_path,
        state_config=state_config_path,
        preload_dir=preload_dir,
        no_preload=no_preload,
        log_level=log_level,
        log_file=log_file,
        debug=debug,
    )

    setup_logging(opts.log_level, opts.log_file)
    if opts.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = options_to_processing_config(opts)

    from brainbuilding.service.eeg_service import EEGService

    pipeline_config, pretrained = load_pipeline_from_yaml(
        opts.pipeline_config,
        preload_dir=(None if opts.no_preload else opts.preload_dir),
        enable_preload=not opts.no_preload,
    )

    service = EEGService(
        pipeline_config,
        tcp_host=config.tcp_host,
        tcp_port=config.tcp_port,
        tcp_retries=config.tcp_retries,
        state_config_path=opts.state_config,
        session_id=opts.session_id,
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
) -> None:
    opts = TrainOptions(
        calibrated_array_path=calibrated_array_path,
        save_calibrated=save_calibrated,
        use_calibrated=use_calibrated,
        sessions_root=sessions_root,
        sessions_dirs=sessions_dirs,
        output_dir=output_dir,
        exclude_label=exclude_label,
        pipeline_config=pipeline_config_path,
        state_config=state_config_path,
        preload_dir=preload_dir,
        no_preload=no_preload,
    )

    setup_logging("INFO", None)

    pipeline_config, _ = load_pipeline_from_yaml(
        opts.pipeline_config,
        preload_dir=(None if opts.no_preload else opts.preload_dir),
        enable_preload=not opts.no_preload,
    )

    if not opts.use_calibrated:
        from brainbuilding.service.eeg_service import EEGOfflineRunner

        if opts.sessions_dirs is not None and len(opts.sessions_dirs) > 0:
            session_dirs = [d for d in opts.sessions_dirs if os.path.isdir(d)]
        else:
            session_dirs = [
                os.path.join(opts.sessions_root, d)
                for d in os.listdir(opts.sessions_root)
                if os.path.isdir(os.path.join(opts.sessions_root, d))
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
                state_config_path=opts.state_config,
            )
            rows = runner.run_from_xdf(xdf_path, session_id=sess_id)
            all_rows.extend(rows)

        data = np.concatenate(all_rows) if len(all_rows) > 0 else np.array([])
        if data.size == 0:
            raise RuntimeError("No calibrated data produced from provided sessions")

        if opts.save_calibrated:
            np.save(opts.calibrated_array_path, data)
        all_rows_arr = data
    else:
        all_rows_arr = np.load(opts.calibrated_array_path)

    logging.info("%s", f"{all_rows_arr.shape=}")
    logging.info("%s", f"{all_rows_arr['sample'].shape=}")

    y = all_rows_arr["label"]
    training_cfg = pipeline_config.training_only_config()
    fitted = training_cfg.fit_training(all_rows_arr, y)

    os.makedirs(opts.output_dir, exist_ok=True)
    for name, comp in fitted.items():
        outp = os.path.join(opts.output_dir, f"{name}.joblib")
        joblib_dump(comp, outp)
    logging.info("Saved %d components to %s", len(fitted), opts.output_dir)


def main(argv: Optional[List[str]] = None) -> None:
    app()


if __name__ == "__main__":
    main()
