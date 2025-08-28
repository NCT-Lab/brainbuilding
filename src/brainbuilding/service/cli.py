"""
Command-line interface for the brainbuilding real-time EEG processor.
"""

import argparse
import logging
from typing import Optional

from brainbuilding.config import (
    CHANNELS_IN_STREAM,
    CHANNELS_TO_KEEP,
    DEFAULT_EEG_STREAM_NAME,
    DEFAULT_EVENT_STREAM_NAME,
    DEFAULT_HISTORY_PATH,
    DEFAULT_PIPELINE_PATH,
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
        "--pipeline-path",
        type=str,
        default=DEFAULT_PIPELINE_PATH,
        help="Path to the trained pipeline pickle file",
    )
    file_group.add_argument(
        "--history-path",
        type=str,
        default=DEFAULT_HISTORY_PATH,
        help="Path to save processing history",
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
        pipeline_path=args.pipeline_path,
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
