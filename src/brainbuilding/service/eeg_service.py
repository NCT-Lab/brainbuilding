import multiprocessing as mp
import time
import logging
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Type
from enum import IntEnum

import numpy as np
from pylsl import StreamInlet, resolve_stream  # type: ignore

from brainbuilding.service.signal import OnlineSignalFilter
from brainbuilding.service.tcp_sender import TCPSender
from brainbuilding.core.config import (
    DEFAULT_TCP_HOST,
    DEFAULT_TCP_PORT,
    DEFAULT_TCP_RETRIES,
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
    pipeline_type: str
    processing_time: float
    window_metadata: WindowMetadata


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
    window_data: np.ndarray,
    fitted_components: Dict[str, Any],
    pipeline_config: Any,
    window_metadata: WindowMetadata,
) -> Optional[ClassificationResult]:
    """Pure inference pipeline with unified transformer/classifier interface"""
    start_time = time.time()

    data = window_data
    try:
        for step in pipeline_config.steps:
            component = fitted_components[step.name]
            wrapped_component = step.wrap_component(component)
            data = wrapped_component.transform(data)
        processing_time = time.time() - start_time
        # Expect classifier wrapper to output [pred, proba]
        if (
            isinstance(data, np.ndarray)
            and data.ndim == 1
            and data.shape[0] >= 2
        ):
            prediction = int(data[0])
            probability = float(data[1])
        else:
            prediction = (
                int(data[0]) if isinstance(data, np.ndarray) else int(data)
            )
            probability = (
                float(data[1])
                if (hasattr(data, "__len__") and len(data) > 1)
                else 1.0
            )
        LOG_INFER.info(
            "Prediction=%s Prob=%.3f Time=%.3fs",
            prediction,
            probability,
            processing_time,
        )
        return ClassificationResult(
            prediction=prediction,
            probability=probability,
            pipeline_type="hybrid",
            processing_time=processing_time,
            window_metadata=window_metadata,
        )
    except KeyError as e:
        processing_time = time.time() - start_time
        LOG_INFER.warning("Inference failed: missing component %s", e)
        return None
    except ValueError as e:
        processing_time = time.time() - start_time
        LOG_INFER.warning("Inference failed: %s", e)
        return None


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

    def process_samples_chunk(
        self, samples: list[np.ndarray]
    ) -> list[np.ndarray]:
        return [self.process(sample) for sample in samples]


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
    window_action_executor: Dict[TransitionAction, Callable[[], None]]
    window_action_callbacks: Dict[
        TransitionAction,
        Callable[[IntEnum], Callable[[np.ndarray, List[float]], None]],
    ]
    action_handlers: Dict[TransitionAction, Callable]
    
    # TODO: надо бы сбрасывать когда сбрасываем логическую группу
    processed_windows_during_state_so_far: int = 0
    current_window_step: int = 0

    def __init__(
        self,
        pipeline_config,
        executor: ProcessPoolExecutor,
        queue: mp.Queue,
        state_config_path: str,
        session_id: Optional[str] = None,
    ):
        self.pipeline_config = pipeline_config
        self.executor = executor

        self.processed_samples: List[np.ndarray] = []
        self.samples_timestamps: List[float] = []

        self.grouped_samples: Dict[IntEnum, List[np.ndarray]] = {}
        self.grouped_timestamps: Dict[IntEnum, List[float]] = {}

        self.fitted_components: Dict[str, Any] = {}
        self.pipeline_ready = False

        self.last_processed_index = 0
        self.window_id = 0
        self.queue = queue

        self.collected_windows: List[ProcessedWindow] = []

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

        # Map windowed actions to their executors (unified)
        self.window_action_executor = {
            TransitionAction.PARTIAL_FIT: (
                lambda: self._maybe_process_windowed_action(
                    TransitionAction.PARTIAL_FIT
                )
            ),
            TransitionAction.COLLECT_FOR_TRAIN: (
                lambda: self._maybe_process_windowed_action(
                    TransitionAction.COLLECT_FOR_TRAIN
                )
            ),
            TransitionAction.PREDICT: (
                lambda: self._maybe_process_windowed_action(
                    TransitionAction.PREDICT
                )
            ),
        }

        self._session_id = session_id
        self._state_visit_counter = 0

        runtime: StateMachineRuntime = load_state_machine_from_yaml(
            state_config_path
        )
        self.processing_state = runtime.processing_state_enum
        self.logical_group = runtime.logical_group_enum
        self.states = runtime.states
        self.current_state = list(runtime.processing_state_enum)[0]
        self._state_visit_counter = 0
        # Initialize grouped buffers for dynamic groups
        self.grouped_samples = {group: [] for group in self.logical_group}
        self.grouped_timestamps = {
            group: [] for group in self.logical_group
        }
        self.partial_fit_last_index = {
            group: 0 for group in self.logical_group
        }
        self.group_segment_start = {group: 0 for group in self.logical_group}

        # Strictly typed windowing from runtime
        self.window_action_window_size = runtime.window_size
        self.window_action_step_size: Dict[TransitionAction, int] = {
            action: int(step) for action, step in runtime.step_size_by_action.items()
        }

        # Submitters and callbacks for windowed actions (data-driven)
        self.submit_map: Dict[
            TransitionAction,
            Callable[[np.ndarray, List[float], WindowMetadata, IntEnum], Future],
        ] = {
            TransitionAction.PARTIAL_FIT: lambda data, ts, meta, group: self.executor.submit(
                self.pipeline_config.partial_fit_components,
                data,
                self.fitted_components,
            ),
            TransitionAction.COLLECT_FOR_TRAIN: lambda data, ts, meta, group: self.executor.submit(
                lambda w, t, m, g: ProcessedWindow(
                    data=w,
                    timestamps=t,
                    metadata=m,
                    class_label=int(g),
                ),
                data,
                ts,
                meta,
                group,
            ),
            TransitionAction.PREDICT: lambda data, ts, meta, group: self.executor.submit(
                process_window_in_pool,
                data,
                self.fitted_components.copy(),
                self.pipeline_config,
                meta,
            ),
        }
        self.callback_map: Dict[TransitionAction, Callable[[Future], None]] = {
            TransitionAction.PARTIAL_FIT: self._handle_components_update,
            TransitionAction.COLLECT_FOR_TRAIN: self._handle_collect_result,
            TransitionAction.PREDICT: self._handle_prediction_result,
        }

        # Action handlers (instance-bound)
        self.action_handlers = {
            TransitionAction.DO_NOTHING: lambda group: None,
            TransitionAction.FIT: self._handle_fit,
            TransitionAction.PARTIAL_FIT: (
                lambda group: self._handle_window_action(
                    TransitionAction.PARTIAL_FIT, group
                )
            ),
            TransitionAction.PREDICT: (
                lambda group: self._handle_window_action(
                    TransitionAction.PREDICT, group
                )
            ),
            TransitionAction.COLLECT_FOR_TRAIN: (
                lambda group: self._handle_window_action(
                    TransitionAction.COLLECT_FOR_TRAIN, group
                )
            ),
        }

    def handle_events(self, events: List[int]):
        for event in events:
            current_state_def = self.states[self.current_state]

            if event in current_state_def.accepted_events:
                next_state = current_state_def.accepted_events[event]
                self._transition_to(next_state)
            else:
                LOG_STATE.debug(
                    "Event %s not accepted in state %s",
                    event,
                    self.current_state.name,
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
            self.action_handlers[action](group)

    def add_new_samples(
        self, samples: List[np.ndarray], timestamps: List[float]
    ):
        self.processed_samples.extend(samples)
        self.samples_timestamps.extend(timestamps)

        current_state_def = self.states[self.current_state]
        if current_state_def.data_collection_group:
            group = current_state_def.data_collection_group
            # Start a new segment if group switched
            if self.current_collection_group != group:
                self.group_segment_start[group] = len(self.grouped_samples[group])
                # Reset per-action last indices to segment start
                for action in TransitionAction:
                    self.window_action_last_index[action][group] = self.group_segment_start[group]
                self.current_collection_group = group
            self.grouped_samples[group].extend(samples)
            self.grouped_timestamps[group].extend(timestamps)

        for action, active in self.window_action_active.items():
            if not active:
                continue
            executor = self.window_action_executor.get(action)
            if executor is not None:
                executor()

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
        start_index: int,
        window_size: int,
        step_size: int,
        fn: Callable[[np.ndarray, List[float]], None],
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
            window_data = np.array(window_slice).T[None, :, :]
            fn(window_data, window_ts)
            start_index += step_size
        return start_index

    def _handle_fit(self, data_group):
        if not self.grouped_samples[data_group]:
            LOG_STATE.error(
                "No data available for fitting from %s", data_group.name
            )
            return

        group_data = np.array(self.grouped_samples[data_group]).T[
            None, :, :
        ]  # (n_times, n_channels) -> (1, n_channels, n_times)
        op_id = f"fit_{data_group.name}"

        # Fit only the next unfitted stateful step in pipeline order
        future = self.executor.submit(
            self.pipeline_config.fit_next_stateful,
            group_data,
            self.fitted_components,
        )
        future.add_done_callback(self._handle_components_update)
        LOG_STATE.info(
            "Submitted async fit %s with %d samples from %s",
            op_id,
            len(group_data),
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

        status = "enabled" if new_active else "disabled"
        suffix = (
            f" for {data_group.name}" if data_group is not None else ""
        )
        LOG_STATE.info("%s %s%s", action.name, status, suffix)

    def _handle_prediction_result(self, future: Future):
        """Callback for completed prediction futures."""
        result = future.result()
        if result is None:
            LOG_STATE.debug("Inference skipped for this window")
            return
        if self.queue:
            self.queue.put(asdict(result))

        LOG_STATE.info(
            "Window %s: pred=%s prob=%.3f time=%.3fs",
            result.window_metadata.window_id,
            result.prediction,
            result.probability,
            result.processing_time,
        )

    def _maybe_process_windowed_action(self, action: TransitionAction):
        group = self.window_action_group[action]
        if group is None:
            return
        # Constrain to current segment to avoid cross-boundary windows
        segment_start = self.group_segment_start.get(group, 0)
        samples_buffer = self.grouped_samples[group][segment_start:]
        timestamps_buffer = self.grouped_timestamps[group][segment_start:]
        start_index = max(
            self.window_action_last_index[action].get(group, 0) - segment_start,
            0,
        )
        window_size = self.window_action_window_size
        step_size = self.window_action_step_size[action]

        def per_window_cb(window_data: np.ndarray, ts: List[float]):
            self._handle_window_step(action, group, window_data, ts)

        new_start = self._for_each_window(
            samples_buffer,
            timestamps_buffer,
            start_index,
            window_size,
            step_size,
            per_window_cb,
        )
        self.window_action_last_index[action][group] = segment_start + new_start

    def _handle_window_step(
        self,
        action: TransitionAction,
        group: IntEnum,
        window_data: np.ndarray,
        ts: List[float],
    ) -> None:
        window_metadata = WindowMetadata(
            start_timestamp=ts[0],
            end_timestamp=ts[-1],
            window_id=f"window_{self.window_id}",
            session_id=self._session_id,
            state_visit_id=self._state_visit_counter,
        )
        future = self.submit_map[action](window_data, ts, window_metadata, group)
        future.add_done_callback(self.callback_map[action])
        self.window_action_counter[action] += 1
        self.window_id += 1

    def _handle_components_update(self, future: Future):
        result = future.result()
        LOG_STATE.info("Fitted components updated: %s", list(result.keys()))
        self.fitted_components.update(result)
        self.pipeline_ready = self._have_all_components()

    def _handle_collect_result(self, future: Future):
        window = future.result()
        self.collected_windows.append(window)


class EEGService:
    def __init__(
        self,
        pipeline_config,
        tcp_host: str | None = None,
        tcp_port: int | None = None,
        tcp_retries: int | None = None,
        state_config_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.stream_manager = StreamManager()
        self.executor = ProcessPoolExecutor()
        self.point_processor = PointProcessor(
            sfreq=250, n_channels=len(CHANNELS_TO_KEEP)
        )
        self.output_queue: mp.Queue = mp.Queue()
        self.state_manager = StateManager(
            pipeline_config,
            self.executor,
            self.output_queue,
            state_config_path,
            session_id,
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
