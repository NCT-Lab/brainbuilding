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
from brainbuilding.pipelines import (
    PipelineConfig,
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

    # Preload models
    preload_group = parser.add_argument_group("Model Preloading")
    preload_group.add_argument(
        "--preload-dir",
        type=str,
        default="models",
        help="Directory containing preloaded components (pt.pkl optional), csp.json, classifier.joblib",
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
        help="Train PT/CSP/SVC models from a dataset and save to models directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw-dataset-2",
        help="Path to raw data directory with per-subject subfolders (Task.json, data.xdf)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained components: pt.joblib, csp.json, classifier.joblib",
    )
    train_parser.add_argument(
        "--exclude-label",
        type=int,
        default=2,
        help="Exclude samples with this label from training (set to -1 to disable)",
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


def _load_pretrained_components(preload_dir: str) -> dict:
    """Load pre-trained PT (optional), CSP and SVC classifier from disk."""
    components = {}

    for fname in ("pt.pkl", "pt.joblib"):
        path = os.path.join(preload_dir, fname)
        if os.path.exists(path):
            try:
                components["pt"] = joblib_load(path)
            except Exception:
                pass
            break

    for fname in ("csp.pkl", "csp.joblib"):
        path = os.path.join(preload_dir, fname)
        if os.path.exists(path):
            components["csp"] = joblib_load(path)
            break
    else:
        csp_json = os.path.join(preload_dir, "csp.json")
        if os.path.exists(csp_json):
            from brainbuilding.transformers import StructuredCSP
            try:
                import json
                with open(csp_json, "r") as f:
                    state = json.load(f)
                filters = np.array(state.get("filters", []), dtype=float)
                patterns = np.array(state.get("patterns", []), dtype=float)
                n_components = int(state.get("n_components", filters.shape[0]))
                csp = StructuredCSP(field='sample', nfilter=n_components, log=False)
                # Restore learned parameters
                csp.filters_ = filters
                csp.patterns_ = patterns
                components["csp"] = csp
            except Exception:
                pass

    clf_path = os.path.join(preload_dir, "classifier.joblib")
    if os.path.exists(clf_path):
        components["classifier"] = joblib_load(clf_path)

    return components


def _create_training_steps():
    """Create pipeline steps matching the evaluation PT pipeline.

    This sequence yields PT→CSP→SVC on top of whitening, background filtering,
    and covariance estimation so that the exported components align with
    what the realtime service expects to preload (pt, csp, classifier).
    """
    return WHITENING + BACKGROUND_FILTER + AUGMENTED_OAS_COV + PT_CSP_LOG + SVC_CLASSIFICATION


def _export_csp_to_json(csp, out_path: str) -> None:
    """Export a fitted CSP to JSON (filters, patterns, n_components)."""
    try:
        import json
        state = {
            "filters": np.asarray(getattr(csp, "filters_", []), dtype=float).tolist(),
            "patterns": np.asarray(getattr(csp, "patterns_", []), dtype=float).tolist(),
            "n_components": int(getattr(csp, "nfilter", 0)),
        }
        with open(out_path, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logging.warning(f"Failed to export CSP to JSON at {out_path}: {e}")


def _train_and_save_models(dataset_path: str, output_dir: str, exclude_label: int = 2) -> None:
    """Train PT/CSP/SVC from a structured dataset and persist artifacts.

    Expected dataset dtype fields: 'sample', 'label', 'subject_id', 'is_background', 'event_id'.
    """
    logging.info(f"Loading dataset from {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    X = data
    y = data["label"]

    if exclude_label >= 0:
        mask = y != exclude_label
        X = X[mask]
        y = y[mask]
        logging.info(f"Excluded label {exclude_label}: remaining samples={len(y)}")

    steps = _create_training_steps()
    pipeline = OnlinePipeline(steps)

    logging.info("Fitting training pipeline (whitening → cov → PT → CSP → SVC)...")
    pipeline.fit(X, y)

    # Extract fitted components
    pt = pipeline.named_steps.get("pt")
    csp = pipeline.named_steps.get("csp")
    clf = pipeline.named_steps.get("svc")

    if pt is None or csp is None or clf is None:
        raise RuntimeError("Training pipeline did not produce required components (pt, csp, svc)")

    os.makedirs(output_dir, exist_ok=True)

    # Save PT
    pt_path = os.path.join(output_dir, "pt.joblib")
    joblib_dump(pt, pt_path)
    logging.info(f"Saved PT to {pt_path}")

    # Save CSP (both joblib and JSON for compatibility with realtime loader)
    csp_joblib_path = os.path.join(output_dir, "csp.joblib")
    joblib_dump(csp, csp_joblib_path)
    logging.info(f"Saved CSP to {csp_joblib_path}")

    csp_json_path = os.path.join(output_dir, "csp.json")
    _export_csp_to_json(csp, csp_json_path)
    logging.info(f"Saved CSP JSON to {csp_json_path}")

    # Save classifier
    clf_path = os.path.join(output_dir, "classifier.joblib")
    joblib_dump(clf, clf_path)
    logging.info(f"Saved classifier to {clf_path}")


def _load_subject_raw(raw_dir: str, subject_id: int):
    """Load MNE Raw for a subject using pyneurostim."""
    try:
        import pyneurostim as ns
    except Exception as e:
        raise RuntimeError("pyneurostim and mne are required for raw-data training") from e

    task_file = os.path.join(raw_dir, str(subject_id), "Task.json")
    xdf_file = os.path.join(raw_dir, str(subject_id), "data.xdf")
    if not (os.path.exists(task_file) and os.path.exists(xdf_file)):
        raise FileNotFoundError(f"Missing Task.json or data.xdf for subject {subject_id} in {raw_dir}")

    from brainbuilding.config import DEFAULT_EEG_STREAM_NAMES, DEFAULT_EVENT_STREAM_NAME

    protocol = ns.io.NeuroStim(
        task_file=task_file,
        xdf_file=xdf_file,
        event_stream_name=DEFAULT_EVENT_STREAM_NAME,
    )
    raw, _ = protocol.raw_xdf(
        annotation=True,
        eeg_stream_names=list(DEFAULT_EEG_STREAM_NAMES),
        extended_annotation=True,
    )
    return raw


def _prepare_raw_channels(raw):
    """Adjust channels to standard names used in config (drop dup, rename)."""
    try:
        raw.drop_channels(["Fp1-1"], on_missing='ignore')  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        raw.rename_channels({"Fp1-0": "Fp1"})  # type: ignore[attr-defined]
    except Exception:
        pass
    return raw


def _preprocess_continuous_online_like(raw):
    """Apply reference subtraction, scaling, channel selection, and bandpass filtering."""
    import numpy as _np
    from brainbuilding.config import CHANNELS_IN_STREAM, REF_CHANNEL_IND, CHANNELS_MASK
    from brainbuilding.service.signal import OnlineSignalFilter

    data = raw.get_data()
    sfreq = raw.info['sfreq']

    # Map configured stream channel order to indices in raw
    ch_name_to_idx = {ch: i for i, ch in enumerate(raw.info['ch_names'])}
    try:
        indices = _np.array([ch_name_to_idx[ch] for ch in CHANNELS_IN_STREAM])
    except KeyError as e:
        missing = [ch for ch in CHANNELS_IN_STREAM if ch not in ch_name_to_idx]
        raise RuntimeError(f"Missing required channels in raw: {missing}") from e

    stream_data = data[indices]  # shape: (len(CHANNELS_IN_STREAM), n_times)
    # Reference subtract Fz
    referenced = stream_data - stream_data[REF_CHANNEL_IND:REF_CHANNEL_IND+1]
    # Convert µV to V
    scaled = referenced / 1_000_000
    # Keep required channels
    selected = scaled[CHANNELS_MASK]

    # Stateful bandpass filter
    filt = OnlineSignalFilter(sfreq=sfreq, n_channels=selected.shape[0])
    filtered_samples = []
    for i in range(selected.shape[1]):
        filtered_samples.append(filt.process_sample(selected[:, i]))
    processed = _np.array(filtered_samples).T  # (n_channels_keep, n_times)
    return processed, sfreq


def _extract_epochs_from_raw(raw, subject_id: int):
    """Create sliding-window epochs and labels from MNE Raw annotations."""
    import numpy as _np
    import mne  # type: ignore
    from brainbuilding.config import (
        STANDARD_EVENT_NAME_TO_ID_MAPPING,
        BACKGROUND_CLASS_INDEX,
        IMAGERY_CLASS_INDEX,
    )

    # Preprocess continuous data
    processed, sfreq = _preprocess_continuous_online_like(raw)

    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    events_reprocessed = []
    for event in events:
        ts_i, _, e = event
        descriptions = [k for k, v in event_ids.items() if v == e]
        desc = descriptions[0] if descriptions else ''
        label_vec = STANDARD_EVENT_NAME_TO_ID_MAPPING(desc.split("#")[0])
        if label_vec == []:
            continue
        events_reprocessed.append((ts_i, 1.0, label_vec))

    # Sliding windows
    sample_delay = int(1.0 * sfreq)
    sample_window = int(3.0 * sfreq)
    sample_step = int(3.0 * sfreq)

    epochs_data = []
    labels = []
    is_background = []
    for ind, (ts_idx, _vas, event_id) in enumerate(events_reprocessed[:-1]):
        if event_id == []:
            continue
        start_idx = ts_idx + sample_delay
        # find next different event
        next_events = [i for i in events_reprocessed[ind+1:] if tuple(i[2]) != tuple(event_id)]
        if not next_events:
            continue
        end_idx = next_events[0][0]

        chunk_start = start_idx
        chunk_end = chunk_start + sample_window
        while chunk_end <= end_idx and (chunk_end <= processed.shape[1]):
            chunk = processed[:, chunk_start:chunk_end]
            epochs_data.append(chunk)
            labels.append(event_id)
            is_background.append(int(event_id[BACKGROUND_CLASS_INDEX] == 1))
            chunk_start = chunk_start + sample_step
            chunk_end = chunk_start + sample_window

    X = _np.array(epochs_data)
    y = _np.array(labels)[:, IMAGERY_CLASS_INDEX]
    is_background_arr = _np.array(is_background)
    subject_ids_arr = _np.full(len(y), subject_id, dtype=_np.int64)

    # Build structured array expected by pipeline
    dtype = [
        ('sample', _np.float64, X.shape[1:]),
        ('label', _np.int64),
        ('subject_id', _np.int64),
        ('is_background', _np.bool_),
        ('event_id', _np.uint64),
    ]
    data = _np.zeros(len(X), dtype=dtype)
    data['sample'] = X
    data['label'] = y
    data['subject_id'] = subject_ids_arr
    data['is_background'] = is_background_arr
    # simple unique ids
    data['event_id'] = _np.arange(len(X), dtype=_np.uint64)
    return data


def _build_dataset_from_raw_dir(raw_dir: str):
    """Load all subject folders under raw_dir and build a single structured array."""
    import numpy as _np
    subject_dirs = [d for d in os.listdir(raw_dir) if d.isdigit()]
    subject_ids = sorted([int(s) for s in subject_dirs])
    all_data = []
    for sid in subject_ids:
        raw = _load_subject_raw(raw_dir, sid)
        raw = _prepare_raw_channels(raw)
        raw.resample(250, npad="auto")  # match DEFAULT_SFREQ
        subject_data = _extract_epochs_from_raw(raw, sid)
        if len(subject_data) > 0:
            all_data.append(subject_data)

    if not all_data:
        raise RuntimeError(f"No valid data found under {raw_dir}")

    data = _np.concatenate(all_data)
    X = data
    y = data['label']
    return X, y


def main(argv=None):
    args = parse_args(argv)

    # If running training subcommand, use its own minimal logging setup
    if getattr(args, "command", None) == "train":
        setup_logging("INFO", None)
        # Always build from raw-data-dir
        X, y = _build_dataset_from_raw_dir(args.raw_data_dir)
        steps = _create_training_steps()
        pipeline = OnlinePipeline(steps)
        logging.info("Fitting training pipeline (whitening → cov → PT → CSP → SVC) from raw data...")
        y_non_background = X['label'][X['is_background'] == 0]
        pipeline.fit(X, y_non_background)

        # Extract fitted components and save
        pt = pipeline.named_steps.get("pt")
        csp = pipeline.named_steps.get("csp")
        clf = pipeline.named_steps.get("svc")
        if pt is None or csp is None or clf is None:
            raise RuntimeError("Training pipeline did not produce required components (pt, csp, svc)")

        os.makedirs(args.output_dir, exist_ok=True)
        joblib_dump(pt, os.path.join(args.output_dir, "pt.joblib"))
        joblib_dump(csp, os.path.join(args.output_dir, "csp.joblib"))
        _export_csp_to_json(csp, os.path.join(args.output_dir, "csp.json"))
        joblib_dump(clf, os.path.join(args.output_dir, "classifier.joblib"))
        logging.info(f"Saved models to {args.output_dir}")
        return

    # Default: serve realtime processor
    setup_logging(args.log_level, args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = args_to_config(args)

    # Lazy import to avoid pylsl dependency during training
    from brainbuilding.service.eeg_service import EEGService
    pipeline_config = PipelineConfig.hybrid_pipeline()
    service = EEGService(
        pipeline_config,
        tcp_host=config.tcp_host,
        tcp_port=config.tcp_port,
        tcp_retries=config.tcp_retries,
    )

    if not args.no_preload:
        pretrained = _load_pretrained_components(args.preload_dir)
        if pretrained:
            service.state_manager.fitted_components.update(pretrained)
            service.state_manager.pipeline_ready = service.state_manager._have_all_components()
            logging.info(
                f"Preloaded components: {list(pretrained.keys())}; pipeline_ready={service.state_manager.pipeline_ready}"
            )

    try:
        service.run()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()

if __name__ == "__main__":
    main()
