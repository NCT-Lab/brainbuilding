"""
Command-line interface for the brainbuilding real-time EEG processor.
"""

import argparse
import logging
from typing import Optional

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
from brainbuilding.train.pipelines import (
    WHITENING,
    BACKGROUND_FILTER,
    AUGMENTED_OAS_COV,
    PT_CSP_LOG,
    SVC_CLASSIFICATION,
    OnlinePipeline,
)
from joblib import load as joblib_load  # type: ignore
from joblib import dump as joblib_dump  # type: ignore
import numpy as np
import os

from brainbuilding.service.pipeline import load_pipeline_from_yaml


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
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


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with idiomatic arguments"""
    parser = argparse.ArgumentParser(
        prog="brainbuilding-realtime",
        description="Real-time EEG processing for brain-computer interfaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Subcommands: default (serve) and train
    subparsers = parser.add_subparsers(dest="command")

    # -------------------------
    # Serve subcommand (default)
    # -------------------------
    stream_group = parser.add_argument_group("Stream Configuration")
    stream_group.add_argument(
        "--eeg-stream",
        dest="eeg_stream_name",
        default=DEFAULT_EEG_STREAM_NAME,
        help="Name of the EEG LSL stream",
    )
    stream_group.add_argument(
        "--event-stream",
        dest="event_stream_name",
        default=DEFAULT_EVENT_STREAM_NAME,
        help="Name of the event LSL stream",
    )

    channel_group = parser.add_argument_group("Channel Configuration")
    channel_group.add_argument(
        "--channels",
        nargs="+",
        default=list(CHANNELS_IN_STREAM),
        help="List of available channels in the stream",
    )
    channel_group.add_argument(
        "--channels-to-keep",
        nargs="+",
        default=list(CHANNELS_TO_KEEP),
        help="List of channels to keep for processing",
    )
    channel_group.add_argument(
        "--reference-channel",
        default=REFERENCE_CHANNEL,
        help="Reference channel for re-referencing",
    )

    processing_group = parser.add_argument_group("Processing Parameters")
    processing_group.add_argument(
        "--window-size",
        dest="processing_window_size",
        type=int,
        default=DEFAULT_PROCESSING_WINDOW_SIZE,
        help="Window size in samples for classification",
    )
    processing_group.add_argument(
        "--processing-step",
        type=int,
        default=DEFAULT_PROCESSING_STEP,
        help="Step size in samples between classifications",
    )
    processing_group.add_argument(
        "--sfreq",
        type=float,
        default=DEFAULT_SFREQ,
        help="Sampling frequency in Hz",
    )
    processing_group.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session identifier (online run)",
    )
    processing_group.add_argument(
        "--scale-factor",
        type=float,
        default=DEFAULT_SCALE_FACTOR,
        help="Scale factor for unit conversion (default: µV to V)",
    )

    # TCP configuration
    tcp_group = parser.add_argument_group("TCP Configuration")
    tcp_group.add_argument(
        "--tcp-host",
        default=DEFAULT_TCP_HOST,
        help="TCP host for sending results",
    )
    tcp_group.add_argument(
        "--tcp-port",
        type=int,
        default=DEFAULT_TCP_PORT,
        help="TCP port for sending results",
    )
    tcp_group.add_argument(
        "--tcp-retries",
        type=int,
        default=DEFAULT_TCP_RETRIES,
        help="Maximum number of TCP connection retries",
    )

    # File paths
    file_group = parser.add_argument_group("File Paths")
    file_group.add_argument(
        "--pipeline-config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to YAML pipeline configuration",
    )
    file_group.add_argument(
        "--history-path",
        type=str,
        default=DEFAULT_HISTORY_PATH,
        help="Path to save processing history",
    )
    file_group.add_argument(
        "--state-config",
        type=str,
        default="state_config_example.yaml",
        help="Path to YAML state machine configuration",
    )

    # Logging
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    logging_group.add_argument(
        "--log-file", type=str, help="Path to log file (default: console only)"
    )

    # Preload models
    preload_group = parser.add_argument_group("Model Preloading")
    preload_group.add_argument(
        "--preload-dir",
        type=str,
        default="models",
        help=(
            "Directory containing preloaded components (pt.pkl optional), "
            "csp.json, classifier.joblib"
        ),
    )
    preload_group.add_argument(
        "--no-preload",
        action="store_true",
        help="Disable preloading of PT/CSP/SVC components",
    )
    logging_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # -------------------------
    # Train subcommand
    # -------------------------
    train_parser = subparsers.add_parser(
        "train",
        help=(
            "Train PT/CSP/SVC models from a dataset and save to models directory"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--calibrated-array-path",
        type=str,
        default="data/calibrated.npy",
        # help=(
        #     "Path to raw data directory with per-subject subfolders "
        #     "(Task.json, data.xdf)"
        # ),
    )
    # train_parser.add_argument(
    #     "--save-calibrated",
    #     action="save_calibrated",
    # )
    # train_parser.add_argument(
    #     "--use-calibrated",
    #     action="use_calibrated"
    # )
    train_parser.add_argument(
        "--sessions-root",
        type=str,
        default="data",
        help=(
            "Root directory with per-session subfolders; each subfolder name is the session_id"
            ", containing data.xdf with EEG and Events streams"
        ),
    )
    train_parser.add_argument(
        "--sessions-dirs",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Explicit list of session directories to use (each must contain data.xdf). "
            "If provided, overrides --sessions-root scanning."
        ),
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help=(
            "Directory to save trained components: pt.joblib, csp.json, "
            "classifier.joblib"
        ),
    )
    train_parser.add_argument(
        "--exclude-label",
        type=int,
        default=2,
        help=(
            "Exclude samples with this label from training (set to -1 to disable)"
        ),
    )
    train_parser.add_argument(
        "--pipeline-config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to YAML pipeline configuration for training",
    )
    # Accept state-config after the 'train' subcommand too (not only as a global arg)
    train_parser.add_argument(
        "--state-config",
        type=str,
        default="configs/states/state_config.yaml",
        help="Path to YAML state machine configuration",
    )

    return parser


def args_to_config(args: argparse.Namespace) -> EEGProcessingConfig:
    """Convert parsed arguments to configuration object"""
    return EEGProcessingConfig(
        # Channel configuration
        channels=args.channels,
        reference_channel=args.reference_channel,
        # Processing parameters
        processing_window_size=args.processing_window_size,
        processing_step=args.processing_step,
        sfreq=args.sfreq,
        # Stream configuration
        eeg_stream_name=args.eeg_stream_name,
        event_stream_name=args.event_stream_name,
        # TCP configuration
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        tcp_retries=args.tcp_retries,
        # File paths
        pipeline_path=args.pipeline_config,
        history_path=args.history_path,
        # Misc
        scale_factor=args.scale_factor,
        debug=args.debug,
    )


def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    return parsed_args


# TODO: getattr for commands? refactor this
def main(argv=None):
    args = parse_args(argv)
    args.save_calibrated = True
    args.use_calibrated = False

    # If running training subcommand, use its own minimal logging setup
    if getattr(args, "command", None) == "train":
        setup_logging("INFO", None)
        pipeline_config, _ = load_pipeline_from_yaml(args.pipeline_config)
        if not args.use_calibrated:
            from brainbuilding.service.eeg_service import EEGOfflineRunner

            if args.sessions_dirs:
                session_dirs = [d for d in args.sessions_dirs if os.path.isdir(d)]
            else:
                session_dirs = [
                    os.path.join(args.sessions_root, d)
                    for d in os.listdir(args.sessions_root)
                    if os.path.isdir(os.path.join(args.sessions_root, d))
                ]

            all_rows = []
            for sess_dir in sorted(session_dirs):
                sess_id = os.path.basename(sess_dir)
                xdf_path = os.path.join(sess_dir, "data.xdf")
                if not os.path.exists(xdf_path):
                    print(f"Skipping {xdf_path} because it does not exist")
                    continue
                logging.info("Loading %s", xdf_path)
                runner = EEGOfflineRunner(
                    pipeline_config=pipeline_config,
                    state_config_path=getattr(
                        args, "state_config", "configs/states/state_config.yaml"
                    ),
                )
                rows = runner.run_from_xdf(xdf_path, session_id=sess_id)
                # logging.info(f"{rows[0]['sample'].shape=}")
                all_rows.extend(rows)
    
            data = np.concatenate(all_rows)

            if args.save_calibrated:
                np.save(args.calibrated_array_path, data)
            all_rows = data
        else:
            all_rows = np.load(args.calibrated_array_path)
        
        logging.info(f"{all_rows.shape=}")
        logging.info(f"{all_rows['sample'].shape=}")

        # Prepare arrays (no assumptions about pipeline content)
        y = all_rows['label']

        # Train strictly by YAML (non-calibration steps only)
        training_cfg = pipeline_config.training_only_config()
        fitted = training_cfg.fit_training(all_rows, y)

        os.makedirs(args.output_dir, exist_ok=True)
        # Save all fitted components generically
        for name, comp in fitted.items():
            outp = os.path.join(args.output_dir, f"{name}.joblib")
            joblib_dump(comp, outp)
        logging.info("Saved %d components to %s", len(fitted), args.output_dir)
        return

    # Default: serve realtime processor
    setup_logging(args.log_level, args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = args_to_config(args)

    # Lazy import to avoid pylsl dependency during training
    from brainbuilding.service.eeg_service import EEGService

    pipeline_config, pretrained = load_pipeline_from_yaml(args.pipeline_config)
    service = EEGService(
        pipeline_config,
        tcp_host=config.tcp_host,
        tcp_port=config.tcp_port,
        tcp_retries=config.tcp_retries,
        state_config_path=getattr(args, "state_config", None),
        session_id=getattr(args, "session_id", None),
    )

    if pretrained:
        service.state_manager.fitted_components.update(pretrained)
        service.state_manager.pipeline_ready = (
            service.state_manager._have_all_components()
        )
        logging.info(
            f"Preloaded components (from YAML): {list(pretrained.keys())}; pipeline_ready={service.state_manager.pipeline_ready}"
        )

    try:
        service.run()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()


if __name__ == "__main__":
    main()
