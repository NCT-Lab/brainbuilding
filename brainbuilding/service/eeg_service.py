import multiprocessing as mp
import time
import logging
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Type, Protocol, Tuple
from enum import IntEnum
import json

import numpy as np
from pylsl import StreamInlet, resolve_stream  # type: ignore
import yaml  # type: ignore

from brainbuilding.service.signal import OnlineSignalFilter
from brainbuilding.service.pipeline import PipelineConfig
from brainbuilding.service.tcp_sender import TCPSender
from brainbuilding.core.config import (
    DEFAULT_TCP_HOST,
    DEFAULT_TCP_PORT,
    DEFAULT_TCP_RETRIES,
    DEFAULT_SFREQ,
)


from brainbuilding.service.state_types import TransitionAction, StateDefinition
from brainbuilding.service.state_config import (
    load_state_machine_from_yaml,
    StateMachineRuntime,
)

LOG = logging.getLogger("brainbuilding.service")
LOG_INFER = logging.getLogger("brainbuilding.service.inference")
LOG_STATE = logging.getLogger("brainbuilding.service.state")
LOG_STREAM = logging.getLogger("brainbuilding.service.stream")
LOG_RUNNER = logging.getLogger("brainbuilding.service.runner")


"""StateDefinition is imported from state_types and used directly."""


@dataclass
class WindowMetadata:
    start_timestamp: float
    end_timestamp: float
    window_id: Optional[str] = None
    session_id: Optional[str] = None
    state_visit_id: Optional[int] = None


@dataclass
class ProcessedWindow:
    data: np.ndarray
    timestamps: List[float]
    metadata: WindowMetadata
    features: Optional[np.ndarray] = None
    class_label: Optional[int] = None


@dataclass
class ClassificationResult:
    prediction: int
    probability: float
    processing_time: float
    ground_truth_labels: Optional[List[int]] = None
    ground_truth_label: Optional[float] = None


DEBUG = False

CHANNELS_IN_STREAM = np.array(
    [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "A1",
        "A2",
        "Fz",
        "Cz",
        "Pz",
    ]
)
REF_CHANNEL_IND = [i for i, ch in enumerate(CHANNELS_IN_STREAM) if ch == "Fz"][
    0
]
CHANNELS_TO_KEEP = np.array(
    [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Cz",
        "Pz",
    ]
)
CHANNELS_MASK = np.array(
    [True if ch in CHANNELS_TO_KEEP else False for ch in CHANNELS_IN_STREAM]
)

TRIGGER_CODE_INDEX = 1


def process_window_in_pool(
    sturctured_window_data: np.ndarray,
    fitted_components: Dict[str, Any],
    pipeline_config: PipelineConfig,
) -> Optional[ClassificationResult]:
    """Pure inference pipeline with unified transformer/classifier interface"""
    start_time = time.time()

    ground_truth_labels = sturctured_window_data["ground_truth_labels"][0].tolist()
    ground_truth_label = sturctured_window_data["label"][0].tolist()

    prediction_result = pipeline_config.predict(sturctured_window_data, fitted_components)
    processing_time = time.time() - start_time

    if prediction_result is not None:
        # [[pred, conf]]
        prediction, probability = prediction_result[0]
    else:
        return None

    LOG_INFER.info(
        "Prediction=%s Prob=%.3f Time=%.3fs",
        prediction,
        probability,
        processing_time,
    )
    return ClassificationResult(
        prediction=prediction,
        probability=probability,
        processing_time=processing_time,
        ground_truth_labels=ground_truth_labels,
        ground_truth_label=ground_truth_label
    )

class StreamManager:
    def __init__(self):
        self.eeg_streams = resolve_stream("type", "EEG")
        self.event_streams = resolve_stream("type", "Events")

        if not self.eeg_streams or not self.event_streams:
            raise RuntimeError("Could not find required streams")

        eeg_info = self.eeg_streams[0]
        event_info = self.event_streams[0]
        self.eeg_inlet = StreamInlet(eeg_info)
        self.event_inlet = StreamInlet(event_info)
        LOG_STREAM.info(
            "Selected EEG stream: name=%s type=%s",
            eeg_info.name(),
            eeg_info.type(),
        )
        LOG_STREAM.info(
            "Selected Events stream: name=%s type=%s",
            event_info.name(),
            event_info.type(),
        )

    def pull_events(self):
        events, _ = self.event_inlet.pull_chunk(timeout=0)
        return events

    def pull_samples(self):
        samples, timestamps = self.eeg_inlet.pull_chunk()
        return samples, timestamps


class SynchronousExecutor:
    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        result = fn(*args, **kwargs)
        fut.set_result(result)
        return fut

    def shutdown(self, wait: bool = True) -> None:
        return None


class PointProcessor:
    def __init__(self, sfreq, n_channels):
        self.signal_filter = OnlineSignalFilter(sfreq, n_channels)

    def process(self, sample: np.ndarray) -> np.ndarray:
        """Process a single raw sample"""
        sample = np.array(sample).reshape(1, -1)
        referenced = sample - sample[:, REF_CHANNEL_IND:REF_CHANNEL_IND + 1]
        scaled = referenced / 1_000_000
        selected = scaled[:, CHANNELS_MASK]
        filtered = self.signal_filter.process_sample(selected)
        return filtered
    
    def process_ndarray(self, samples: np.ndarray) -> np.ndarray:
        assert samples.ndim == 2
        assert samples.shape[1] == len(CHANNELS_MASK)
        referenced = samples - samples[:, REF_CHANNEL_IND:REF_CHANNEL_IND + 1]
        scaled = referenced / 1_000_000
        selected = scaled[:, CHANNELS_MASK]
        filtered = self.signal_filter.process_samples_batch(selected)
        return filtered

    def process_samples_chunk(
        self, samples: list[np.ndarray]
    ) -> list[np.ndarray]:
        return [self.process(sample) for sample in samples]


class ExecutorLike(Protocol):
    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future: ...
    def shutdown(self, wait: bool = True) -> None: ...


class StateManager:
    processing_state: Type[IntEnum]
    logical_group: Type[IntEnum]
    states: Dict[IntEnum, StateDefinition]
    current_state: IntEnum
    window_action_active: Dict[TransitionAction, bool]
    window_action_group: Dict[TransitionAction, Optional[IntEnum]]
    window_action_last_index: Dict[TransitionAction, Dict[IntEnum, int]]
    window_action_window_size: int
    window_action_step_size: Dict[TransitionAction, int]
    window_action_delay_samples: Dict[TransitionAction, int]
    window_action_executor: Dict[TransitionAction, Callable[[], None]]
    window_action_callbacks: Dict[
        TransitionAction,
        Callable[[IntEnum], Callable[[np.ndarray, List[float]], None]],
    ]
    action_handlers: Dict[TransitionAction, Callable]
    window_action_enabled_at_index: Dict[TransitionAction, Dict[IntEnum, int]]
    
    # TODO: надо бы сбрасывать когда сбрасываем логическую группу
    processed_windows_during_state_so_far: int = 0
    current_window_step: int = 0

    def __init__(
        self,
        pipeline_config,
        executor: ExecutorLike,
        queue: mp.Queue,
        state_config_path: str,
        session_id: Optional[str] = None,
        process_fn: Callable = process_window_in_pool,
        sfreq: float = DEFAULT_SFREQ,
    ):
        self.pipeline_config = pipeline_config
        self.executor = executor
        self.process_fn = process_fn
        self.sfreq = float(sfreq)

        self.processed_samples: List[np.ndarray] = []
        self.samples_timestamps: List[float] = []
        self.processed_labels: List[int] = []

        self.grouped_samples: Dict[IntEnum, List[np.ndarray]] = {}
        self.grouped_timestamps: Dict[IntEnum, List[float]] = {}
        self.grouped_labels: Dict[IntEnum, List[int]] = {}

        self.fitted_components: Dict[str, Any] = {}
        self.pipeline_ready = False

        self.last_processed_index = 0
        self.window_id = 0
        self.queue = queue

        self.collected_windows: List[ProcessedWindow] = []
        # Collected calibrated samples for offline training
        self.collected_calibrated: List[dict[str, Any]] = []

        # Windowed action runtime state/config
        self.window_action_active: Dict[TransitionAction, bool] = {
            action: False for action in TransitionAction
        }
        self.window_action_group: Dict[TransitionAction, Optional[IntEnum]] = {
            action: None for action in TransitionAction
        }
        self.window_action_last_index: Dict[
            TransitionAction, Dict[IntEnum, int]
        ] = {action: {} for action in TransitionAction}
        # Per-action window counters
        self.window_action_counter: Dict[TransitionAction, int] = {
            action: 0 for action in TransitionAction
        }
        # Track segment starts per logical group to avoid cross-boundary windows
        self.group_segment_start: Dict[IntEnum, int] = {}
        self.current_collection_group: Optional[IntEnum] = None


        self._session_id = session_id
        self._state_visit_counter = 0

        runtime: StateMachineRuntime = load_state_machine_from_yaml(
            state_config_path
        )
        self.processing_state = runtime.processing_state_enum
        self.logical_group = runtime.logical_group_enum
        self.states = runtime.states
        self.current_state = list(runtime.processing_state_enum)[0]
        self.visited_state_names = [(self.states[self.current_state].name, "")]
        self._state_visit_counter = 0
        # Initialize grouped buffers for dynamic groups
        self.grouped_samples = {group: [] for group in self.logical_group}
        self.grouped_timestamps = {
            group: [] for group in self.logical_group
        }
        self.grouped_labels = {group: [] for group in self.logical_group}
        self.partial_fit_last_index = {
            group: 0 for group in self.logical_group
        }
        self.group_segment_start = {group: 0 for group in self.logical_group}

        # Strictly typed windowing from runtime
        self.window_action_window_size = runtime.window_size
        self.window_action_step_size: Dict[TransitionAction, int] = {
            action: int(step)
            for action, step in runtime.step_size_by_action.items()
        }
        # Convert configured per-action delays (seconds) to samples using sfreq
        self.window_action_delay_samples: Dict[TransitionAction, int] = {
            action: int(round(delay_sec * self.sfreq))
            for action, delay_sec in runtime.start_delay_seconds_by_action.items()
        }

        # Submitters and callbacks for windowed actions (data-driven)
        self.submit_map: Dict[
            TransitionAction,
            Callable[[np.ndarray,], Future],
        ] = {
            TransitionAction.PARTIAL_FIT: (
                lambda data: self.executor.submit(
                    self.pipeline_config.partial_fit_components,
                    data,
                    self.fitted_components,
                )
            ),
            TransitionAction.COLLECT_FOR_TRAIN: (
                lambda data: self.executor.submit(
                    lambda data: data,
                    data,
                )
            ),
            TransitionAction.PREDICT: (
                lambda data: self.executor.submit(
                    self.process_fn,
                    data,
                    self.fitted_components.copy(),
                    self.pipeline_config,
                )
            ),
        }
        self.callback_map: Dict[TransitionAction, Callable[[Future], None]] = {
            TransitionAction.PARTIAL_FIT: self._handle_components_update,
            TransitionAction.COLLECT_FOR_TRAIN: self._handle_collect_result,
            TransitionAction.PREDICT: self._handle_prediction_result,
        }

        # Action handlers (instance-bound)
        self.action_handlers = {
            TransitionAction.DO_NOTHING: lambda _a, _b: None,
            # TODO: need to change to _handle_window_action
            TransitionAction.FIT: self._handle_fit,
            TransitionAction.PARTIAL_FIT: self._handle_window_action,
            TransitionAction.PREDICT: self._handle_window_action,
            TransitionAction.COLLECT_FOR_TRAIN: self._handle_window_action,
        }

        # Track when an action was enabled (per group) to enforce startup delay
        self.window_action_enabled_at_index = {
            action: {} for action in TransitionAction
        }

    def handle_events(self, events: List[int]):
        for event in events:
            current_state_def = self.states[self.current_state]
            # TODO: introduce ignored event ids
            if event == -1:
                return
            if event in current_state_def.accepted_events:
                next_state = current_state_def.accepted_events[event]
                self._transition_to(next_state)
                self.visited_state_names.append([(next_state.name, event)])
            else:
                raise ValueError(
                    "Event %s not accepted in state %s.\nStates so far: %s",
                    event,
                    self.current_state.name,
                    self.visited_state_names
                )

    def _transition_to(self, next_state):
        old_state = self.current_state
        old_state_def = self.states[old_state]
        new_state_def = self.states[next_state]

        self._run_actions_for_trigger(
            state_def=old_state_def,
            actions=old_state_def.on_transition_actions.get(next_state, []),
            trigger="transition",
            next_state=next_state,
        )

        self._run_actions_for_trigger(
            state_def=old_state_def,
            actions=old_state_def.on_exit_actions,
            trigger="exit",
        )

        self.current_state = next_state
        self._state_visit_counter += 1

        self._run_actions_for_trigger(
            state_def=new_state_def,
            actions=new_state_def.on_entry_actions,
            trigger="entry",
        )

        LOG_STATE.info("Transition %s -> %s", old_state.name, next_state.name)

    def _resolve_action_group(
        self,
        state_def: StateDefinition,
        action: TransitionAction,
        trigger: str,
        next_state: Optional[IntEnum] = None,
    ):
        default_group = state_def.data_collection_group
        if trigger == "transition":
            by_next = state_def.on_transition_action_groups
            actions_map = by_next.get(next_state, {}) if next_state else {}
            return actions_map.get(action, default_group)

        trigger_map = {
            "entry": state_def.on_entry_action_groups,
            "exit": state_def.on_exit_action_groups,
        }.get(trigger, {})

        return trigger_map.get(action, default_group)

    def _run_actions_for_trigger(
        self,
        state_def: StateDefinition,
        actions: Optional[List[TransitionAction]],
        trigger: str,
        next_state: Optional[IntEnum] = None,
    ) -> None:
        if not actions:
            return
        for action in actions:
            group = self._resolve_action_group(
                state_def=state_def,
                action=action,
                trigger=trigger,
                next_state=next_state,
            )
            self.action_handlers[action](action, group)

    def add_new_samples(
        self, samples: List[np.ndarray], timestamps: List[float]
    ):
        current_state_def = self.states[self.current_state]
        self.processed_samples.extend(samples)
        self.samples_timestamps.extend(timestamps)
        self.processed_labels.extend(
            [current_state_def.class_label] * len(samples)
        )

        group = current_state_def.data_collection_group

        if self.current_collection_group != group and group is not None:
            self.group_segment_start[group] = len(self.grouped_samples[group])
            # Reset per-action last indices to segment start
            for action in TransitionAction:
                self.window_action_last_index[action][group] = self.group_segment_start[group]
        
        self.current_collection_group = group
        
        if group is not None:
            self.grouped_samples[group].extend(samples)
            self.grouped_timestamps[group].extend(timestamps)
            self.grouped_labels[group].extend(
                [current_state_def.class_label] * len(samples)
            )
            for action, active in self.window_action_active.items():
                if not active:
                    continue
                self._maybe_process_windowed_action(action)

    def _have_all_components(self) -> bool:
        """Check if all pipeline steps are available in fitted_components."""
        return all(
            step.name in self.fitted_components
            for step in self.pipeline_config.steps
        )

    def _for_each_window(
        self,
        samples_buffer: List[np.ndarray],
        timestamps_buffer: List[float],
        labels_buffer: List[int],
        start_index: int,
        window_size: int,
        step_size: int,
        fn: Callable[[np.ndarray, List[float], List[int]], None],
    ) -> int:
        """Generic window iterator over buffers with configurable step."""
        if (len(samples_buffer) - start_index) < window_size:
            return start_index
        while (len(samples_buffer) - start_index) >= window_size:
            window_slice = samples_buffer[
                start_index:start_index + window_size
            ]
            window_ts = timestamps_buffer[
                start_index:start_index + window_size
            ]
            window_labels = labels_buffer[
                start_index:start_index + window_size
            ]
            window_data = np.array(window_slice).T[None, :, :]
            fn(window_data, window_ts, window_labels)
            start_index += step_size
        return start_index

    def _handle_fit(self, _action, data_group):
        if not self.grouped_samples[data_group]:
            LOG_STATE.error(
                "No data available for fitting from %s", data_group.name
            )
            return

        # Prepare a single structured window covering the whole available buffer for this group
        samples_all = self.grouped_samples[data_group]
        timestamps_all = self.grouped_timestamps[data_group]
        window_data = np.array(samples_all).T[None, :, :]
        LOG_STATE.info(f"Window data shape: {window_data.shape}")
        metadata = WindowMetadata(
            start_timestamp=timestamps_all[0],
            end_timestamp=timestamps_all[-1],
            window_id=f"fit_{self.window_id}",
            session_id=self._session_id,
            state_visit_id=self._state_visit_counter,
        )
        structured = self._prepare_array_with_metadata(
            window_data, metadata, labels=self.grouped_labels[data_group]
        )
        op_id = f"fit_{data_group.name}"

        # Fit only the next unfitted stateful step in pipeline order
        future = self.executor.submit(
            self.pipeline_config.fit_next_stateful,
            structured,
            self.fitted_components,
        )
        future.add_done_callback(self._handle_components_update)
        LOG_STATE.info(
            "Submitted async fit %s with %d samples from %s",
            op_id,
            len(structured),
            data_group.name,
        )

    def _handle_window_action(
        self, action: TransitionAction, data_group: Optional[IntEnum]
    ) -> None:
        """Unified initializer/toggler for windowed actions.

        No branching by action type; data-driven toggle and setup.
        """
        same_group = (
            self.window_action_group[action] == data_group
        )
        new_active = not (self.window_action_active[action] and same_group)

        self.window_action_active[action] = new_active
        self.window_action_group[action] = data_group if new_active else None
        _ = (
            new_active
            and data_group is not None
            and self.window_action_last_index[action].setdefault(data_group, 0)
        )
        if new_active and data_group is not None:
            # Record absolute enable index to enforce startup delays
            self.window_action_enabled_at_index[action][
                data_group
            ] = len(self.grouped_samples[data_group])
        else:
            # Clear enable index on disable to avoid stale gating
            if data_group is not None and data_group in self.window_action_enabled_at_index[action]:
                del self.window_action_enabled_at_index[action][data_group]

        status = "enabled" if new_active else "disabled"
        suffix = (
            f" for {data_group.name}" if data_group is not None else ""
        )
        LOG_STATE.info("%s %s%s", action.name, status, suffix)

    def _maybe_process_windowed_action(self, action: TransitionAction):
        group = self.window_action_group[action]
        if group is None:
            return
        # Constrain to current segment to avoid cross-boundary windows
        segment_start = self.group_segment_start.get(group, 0)
        samples_buffer = self.grouped_samples[group][segment_start:]
        timestamps_buffer = self.grouped_timestamps[group][segment_start:]
        labels_buffer = self.grouped_labels[group][segment_start:]
        start_index = max(
            self.window_action_last_index[action].get(group, 0) - segment_start,
            0,
        )
        # Enforce startup delay: require enough samples since action enable
        delay_needed = self.window_action_delay_samples.get(action, 0)
        enable_index_abs = self.window_action_enabled_at_index[action].get(group)
        if enable_index_abs is None:
            return
        # Compute how many samples accumulated since enable within current segment
        current_length_abs = len(self.grouped_samples[group])
        effective_enable_index = max(enable_index_abs, segment_start)
        samples_since_enable = current_length_abs - effective_enable_index
        if samples_since_enable < delay_needed:
            return
        window_size = self.window_action_window_size
        step_size = self.window_action_step_size[action]

        def per_window_cb(window_data: np.ndarray, ts: List[float], labels: List[int]):
            self._handle_window_step(action, group, window_data, ts, labels)
        new_start = self._for_each_window(
            samples_buffer,
            timestamps_buffer,
            labels_buffer,
            start_index,
            window_size,
            step_size,
            per_window_cb,
        )
        self.window_action_last_index[action][group] = segment_start + new_start

    def _prepare_array_with_metadata(
        self,
        window_data: np.ndarray,
        metadata: WindowMetadata,
        labels: List[int],
    ):
        dtype = [
            ("sample", window_data.dtype, window_data.shape[1:]),
            ("label", np.int64),
            ("session_id", np.int64),
            ("state_visit_id", np.int64),
            ("window_id", str),
            ("start_timestamp", np.float64),
            ("end_timestamp", np.float64),
            ("ground_truth_labels", np.int64, (len(labels),)),
        ]
        assert window_data.shape[0] == 1
        array_with_metadata = np.zeros(1, dtype=dtype)
        array_with_metadata["sample"][0] = window_data[0]
        array_with_metadata["label"][0] = self.states[self.current_state].class_label
        array_with_metadata["session_id"][0] = int(metadata.session_id)
        array_with_metadata["state_visit_id"][0] = metadata.state_visit_id
        # TODO: there are some issues with saving strings in structured array
        # we need to fix this
        array_with_metadata["window_id"][0] = metadata.window_id
        array_with_metadata["start_timestamp"][0] = metadata.start_timestamp
        array_with_metadata["end_timestamp"][0] = metadata.end_timestamp
        array_with_metadata["ground_truth_labels"][0] = labels

        return array_with_metadata

    def _handle_window_step(
        self,
        action: TransitionAction,
        group: IntEnum,
        window_data: np.ndarray,
        ts: List[float],
        labels: List[int],
    ) -> None:
        window_metadata = WindowMetadata(
            start_timestamp=ts[0],
            end_timestamp=ts[-1],
            # TODO: current approach is invalid, we need unique window ids per each action
            window_id=f"window_{self.window_id}",
            session_id=self._session_id,
            state_visit_id=self._state_visit_counter,
        )
        window_with_metadata = self._prepare_array_with_metadata(
            window_data, window_metadata, labels=labels
        )
        future = self.submit_map[action](window_with_metadata)
        future.add_done_callback(self.callback_map[action])
        self.window_action_counter[action] += 1
        self.window_id += 1

    def _handle_prediction_result(self, future: Future):
        """Callback for completed prediction futures."""
        result = future.result()
        if result is None:
            LOG_STATE.debug("Inference skipped for this window")
            return
        if self.queue:
            self.queue.put(asdict(result))

    def _handle_components_update(self, future: Future):
        result = future.result()
        LOG_STATE.info("Fitted components updated: %s", list(result.keys()))
        self.fitted_components.update(result)
        self.pipeline_ready = self._have_all_components()

    def _handle_collect_result(self, future: Future):
        structured_window = future.result()
        self.collected_windows.append(structured_window)
        calibrated = self.pipeline_config.apply_calibration(
            structured_window, self.fitted_components
        )
        self.collected_calibrated.append(calibrated)


class EEGService:
    def __init__(
        self,
        pipeline_config,
        tcp_host: str | None = None,
        tcp_port: int | None = None,
        tcp_retries: int | None = None,
        state_config_path: Optional[str] = None,
        session_id: Optional[str] = None,
        sfreq: float = 500.,
    ):
        self.stream_manager = StreamManager()
        self.executor = ProcessPoolExecutor()
        # TODO: make sfreq configurable
        self.point_processor = PointProcessor(
            sfreq=sfreq, n_channels=len(CHANNELS_TO_KEEP)
        )
        self.output_queue: mp.Queue = mp.Queue()
        if state_config_path is None:
            raise ValueError("state_config_path cannot be None")
        self.state_manager = StateManager(
            pipeline_config,
            self.executor,
            self.output_queue,
            state_config_path,
            session_id,
            process_fn=process_window_in_pool,
            sfreq=sfreq,
        )
        host = tcp_host if tcp_host is not None else DEFAULT_TCP_HOST
        port = tcp_port if tcp_port is not None else DEFAULT_TCP_PORT
        retries = (
            tcp_retries if tcp_retries is not None else DEFAULT_TCP_RETRIES
        )
        self.tcp_sender = TCPSender(self.output_queue, host, port, retries)

    def run(self):
        """Main processing loop"""
        self.tcp_sender.start()
        while True:
            try:
                events = self.stream_manager.pull_events()
                if events:
                    self.state_manager.handle_events(
                        [e[TRIGGER_CODE_INDEX] for e in events]
                    )
                samples, timestamps = self.stream_manager.pull_samples()
                if samples:
                    self.state_manager.add_new_samples(
                        self.point_processor.process_samples_chunk(samples),
                        timestamps,
                    )
            except KeyboardInterrupt:
                break
        self.stop()

    def stop(self):
        """Clean shutdown of all components"""
        LOG.info("Stopping EEG service...")
        self.tcp_sender.stop()
        self.executor.shutdown(wait=True)
        LOG.info("EEG service stopped")


class EEGOfflineRunner:
    """Offline runner that reuses StateManager logic to process XDF sessions.

    It feeds events and samples into the StateManager, relying on the same
    windowing and callbacks. Calibrated windows are returned from
    state_manager.collected_calibrated after processing.
    """

    def __init__(
        self,
        pipeline_config,
        state_config_path: str,
        sfreq: float
    ):
        self.pipeline_config = pipeline_config
        self.state_config_path = state_config_path
        self.sfreq = sfreq

    def run_from_xdf(self, xdf_path: str, session_id: str) -> list[dict[str, Any]]:
        from pyxdf import load_xdf  # type: ignore
        LOG_STATE.setLevel('ERROR')

        streams, _ = load_xdf(xdf_path)

        def _info_value(stream: dict, key: str) -> str:
            vals = stream.get("info", {}).get(key, [])
            return vals[0] if isinstance(vals, list) and vals else ""

        eeg = next((s for s in streams if _info_value(s, "type") == "EEG"), None)
        evs = next(
            (
                s
                for s in streams
                if _info_value(s, "type") in ("Events", "Markers")
            ),
            None,
        )
        if eeg is None or evs is None:
            raise RuntimeError("EEGOfflineRunner: EEG or Events stream missing")

        with open(self.state_config_path, "r", encoding="utf-8") as f:
            cfg_raw = yaml.safe_load(f)

        event_ids_cfg = cfg_raw["event_ids"]

        # Build executor/manager (training mode uses synchronous executor)
        executor = SynchronousExecutor()
        queue: mp.Queue = mp.Queue()
        sm = StateManager(
            pipeline_config=self.pipeline_config,
            executor=executor,
            queue=queue,
            state_config_path=self.state_config_path,
            session_id=session_id,
            process_fn=process_window_in_pool,
            sfreq=self.sfreq,
        )
        point_processor = PointProcessor(
            sfreq=self.sfreq, n_channels=len(CHANNELS_TO_KEEP)
        )

        # Feed events (codes at index 1 if present) and validate set equality
        codes: list[int] = [i[TRIGGER_CODE_INDEX] for i in evs['time_series']]
        xdf_id_set = set(codes)
        cfg_id_set = set(int(v) for v in event_ids_cfg.values())
        assert xdf_id_set == cfg_id_set, "XDF event IDs do not match state_config event_ids"
        
        processed_time_series = point_processor.process_ndarray(eeg['time_series'])
        all_time_stamps = eeg["time_stamps"].tolist() + evs["time_stamps"].tolist()
        sorted_indices = np.argsort(all_time_stamps)
        all_samples = [('eeg', i) for i in processed_time_series] + [('event', i) for i in evs["time_series"]]

        for sample_id in sorted_indices:
            sample_label, sample = all_samples[sample_id]
            sample_timestamp = all_time_stamps[sample_id]
            if sample_label == 'eeg':
                sm.add_new_samples(sample[None, ...], [sample_timestamp])
            else:
                sm.handle_events([sample[TRIGGER_CODE_INDEX]])

        executor.shutdown(wait=True)
        return list(sm.collected_calibrated)


class StateCheckRunner:
    """Offline runner to validate state machine transitions against a task JSON.

    This runner processes events from an XDF file and compares the state
    machine's state after each transition with the expected state defined in a
    corresponding task.json file. It does not process EEG samples.
    """

    def __init__(self, state_config_path: str):
        self.state_config_path = state_config_path

    def run_check(
        self, xdf_path: str, task_json_path: str
    ) -> Dict[int, Tuple[str, str]]:
        """Run the state check for a single session."""
        from pyxdf import load_xdf  # type: ignore
        LOG_STATE.setLevel('ERROR')

        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        streams, _ = load_xdf(xdf_path)

        def _info_value(stream: dict, key: str) -> str:
            vals = stream.get("info", {}).get(key, [])
            return vals[0] if isinstance(vals, list) and vals else ""

        evs = next(
            (
                s
                for s in streams
                if _info_value(s, "type") in ("Events", "Markers")
            ),
            None,
        )

        if evs is None:
            raise RuntimeError("StateCheckRunner: Events stream missing")

        executor = SynchronousExecutor()
        queue: mp.Queue = mp.Queue()
        sm = StateManager(
            pipeline_config=PipelineConfig(steps=[]),
            executor=executor,
            queue=queue,
            state_config_path=self.state_config_path,
            process_fn=process_window_in_pool,
            sfreq=DEFAULT_SFREQ,
        )

        actual_states = []
        for event_data in evs["time_series"]:
            old_state = sm.current_state
            event_code = int(event_data[TRIGGER_CODE_INDEX])
            sm.handle_events([event_code])
            new_state = sm.current_state
            if new_state is not old_state:
                actual_states.append(new_state.name)

        expected_states = [
            sample["sample_type"] for sample in task_data["samples"]
        ]

        mismatches = {}
        max_len = max(len(expected_states), len(actual_states))
        for i in range(max_len):
            expected = (
                expected_states[i] if i < len(expected_states) else "<no state>"
            )
            actual = actual_states[i] if i < len(actual_states) else "<no state>"
            if expected.lower() != actual.lower():
                mismatches[i] = (expected, actual)

        executor.shutdown(wait=False)
        return mismatches


class EEGEvaluationRunner:
    def __init__(
        self,
        pipeline_config,
        state_config_path: str,
        sfreq: float,
        pretrained_components: Optional[Dict[str, Any]] = None,
    ):
        self.pipeline_config = pipeline_config
        self.state_config_path = state_config_path
        self.sfreq = sfreq
        self.pretrained_components = pretrained_components or {}

    def run_from_xdf(
        self, xdf_path: str, session_id: str
    ) -> Tuple[List[int], List[int], List[float], List[List[float]]]:
        from pyxdf import load_xdf  # type: ignore

        LOG_STATE.setLevel("ERROR")
        LOG_INFER.setLevel("ERROR")

        streams, _ = load_xdf(xdf_path)

        def _info_value(stream: dict, key: str) -> str:
            vals = stream.get("info", {}).get(key, [])
            return vals[0] if isinstance(vals, list) and vals else ""

        eeg = next((s for s in streams if _info_value(s, "type") == "EEG"), None)
        evs = next(
            (
                s
                for s in streams
                if _info_value(s, "type") in ("Events", "Markers")
            ),
            None,
        )
        if eeg is None or evs is None:
            raise RuntimeError("EEGEvaluationRunner: EEG or Events stream missing")

        # Build executor/manager
        executor = SynchronousExecutor()
        queue: mp.Queue = mp.Queue()
        sm = StateManager(
            pipeline_config=self.pipeline_config,
            executor=executor,
            queue=queue,
            state_config_path=self.state_config_path,
            session_id=session_id,
             process_fn=process_window_in_pool,
             sfreq=self.sfreq,
        )
        sm.fitted_components.update(self.pretrained_components)
        sm.pipeline_ready = sm._have_all_components()
        point_processor = PointProcessor(
            sfreq=self.sfreq, n_channels=len(CHANNELS_TO_KEEP)
        )

        processed_time_series = point_processor.process_ndarray(eeg['time_series'])
        all_time_stamps = eeg["time_stamps"].tolist() + evs["time_stamps"].tolist()
        sorted_indices = np.argsort(all_time_stamps)
        all_samples = [('eeg', i) for i in processed_time_series] + [('event', i) for i in evs["time_series"]]

        for sample_id in sorted_indices:
            sample_label, sample = all_samples[sample_id]
            sample_timestamp = all_time_stamps[sample_id]
            if sample_label == 'eeg':
                sm.add_new_samples(sample[None, ...], [sample_timestamp])
            else:
                sm.handle_events([sample[TRIGGER_CODE_INDEX]])

        executor.shutdown(wait=True)

        all_predictions = []
        all_ground_truth = []
        all_probabilities = []
        # all_class_probabilities_agg = []
        while not queue.empty():
            result = ClassificationResult(**queue.get())
            if result.ground_truth_labels:
                window_ground_truth = int(np.round(np.mean(result.ground_truth_labels)))
                all_ground_truth.append(window_ground_truth)
                all_predictions.append(result.prediction)

                all_probabilities.append(result.probability)

                # if result.all_probabilities:
                #     all_class_probabilities_agg.append(result.all_probabilities)

        return all_predictions, all_ground_truth, all_probabilities
