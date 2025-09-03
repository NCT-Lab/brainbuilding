import multiprocessing as mp
import time
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


"""StateDefinition is imported from state_types and used directly."""


@dataclass
class WindowMetadata:
    start_timestamp: float
    end_timestamp: float
    window_id: Optional[str] = None


@dataclass
class ProcessedWindow:
    data: np.ndarray
    timestamps: List[float]
    metadata: WindowMetadata
    features: Optional[np.ndarray] = None


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
        print(
            f"Prediction: {prediction}, Probability: {probability}, Processing Time: {processing_time}"
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
        print(f"Inference failed: missing component {e}")
        return None
    except ValueError as e:
        processing_time = time.time() - start_time
        print(f"Inference failed: {e}")
        return None


class StreamManager:
    def __init__(self):
        self.eeg_streams = resolve_stream("name", "NeoRec21-1247")
        self.event_streams = resolve_stream("name", "Brainbuilding-Events")

        if not self.eeg_streams or not self.event_streams:
            raise RuntimeError("Could not find required streams")

        self.eeg_inlet = StreamInlet(self.eeg_streams[0])
        self.event_inlet = StreamInlet(self.event_streams[0])

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
        referenced = sample - sample[:, REF_CHANNEL_IND : REF_CHANNEL_IND + 1]
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

    def __init__(
        self,
        pipeline_config,
        executor: ProcessPoolExecutor,
        queue: mp.Queue,
        state_config_path: Optional[str] = None,
    ):
        self.pipeline_config = pipeline_config
        self.executor = executor

        self.processed_samples: List[np.ndarray] = []
        self.samples_timestamps: List[float] = []

        self.grouped_samples: Dict[IntEnum, List[np.ndarray]] = {}
        self.grouped_timestamps: Dict[IntEnum, List[float]] = {}

        self.fitted_components: Dict[str, Any] = {}
        self.pending_operations: Dict[str, Future] = {}
        self.pipeline_ready = False

        self.window_size = 250
        self.step_size = 125
        self.inference_active = False
        self.last_processed_index = 0
        self.window_id = 0
        self.queue = queue

        self.partial_fit_active = False
        self.partial_fit_window_size = self.window_size
        self.partial_fit_last_index: Dict[IntEnum, int] = {}
        self._partial_fit_group: Optional[IntEnum] = None

        self.action_handlers: Dict[TransitionAction, Callable] = {
            TransitionAction.DO_NOTHING: lambda group: None,
            TransitionAction.FIT: self._handle_fit,
            TransitionAction.PARTIAL_FIT: self._handle_partial_fit,
            TransitionAction.PREDICT: self._handle_predict,
        }

        if state_config_path:
            runtime: StateMachineRuntime = load_state_machine_from_yaml(
                state_config_path
            )
            self.processing_state = (
                runtime.processing_state_enum
            )  # dynamic enums
            self.logical_group = runtime.logical_group_enum
            self.states = runtime.states
            self.current_state = list(runtime.processing_state_enum)[0]
            # Initialize grouped buffers for dynamic groups
            self.grouped_samples = {group: [] for group in self.logical_group}
            self.grouped_timestamps = {
                group: [] for group in self.logical_group
            }
            self.partial_fit_last_index = {
                group: 0 for group in self.logical_group
            }
        else:
            # Require config; no inline fallback
            raise RuntimeError(
                "State machine YAML config path must be provided"
            )

    def handle_events(self, events: List[int]):
        for event in events:
            current_state_def = self.states[self.current_state]

            if event in current_state_def.accepted_events:
                next_state = current_state_def.accepted_events[event]
                self._transition_to(next_state)
            else:
                print(
                    f"Event '{event}' not accepted in state {self.current_state.name}"
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

        self._run_actions_for_trigger(
            state_def=new_state_def,
            actions=new_state_def.on_entry_actions,
            trigger="entry",
        )

        print(f"State transition: {old_state.name} -> {next_state.name}")

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
        self._check_pending_operations()

        self.processed_samples.extend(samples)
        self.samples_timestamps.extend(timestamps)

        current_state_def = self.states[self.current_state]
        if current_state_def.data_collection_group:
            group = current_state_def.data_collection_group
            self.grouped_samples[group].extend(samples)
            self.grouped_timestamps[group].extend(timestamps)

        if (
            self.inference_active
            and len(self.processed_samples) >= self.window_size
        ):
            self._maybe_process_latest_window()

        if self.partial_fit_active:
            self._maybe_partial_fit_latest_window()

    def _check_pending_operations(self):
        completed_ops = []
        for op_id, future in self.pending_operations.items():
            if future.done():
                result = future.result()
                if op_id.startswith("fit_") or op_id.startswith(
                    "partial_fit_"
                ):
                    self.fitted_components.update(result)
                    self.pipeline_ready = self._have_all_components()
                    print(f"Async operation {op_id} completed successfully")
                completed_ops.append(op_id)

        for op_id in completed_ops:
            del self.pending_operations[op_id]

    def _have_all_components(self) -> bool:
        """Check if all pipeline steps are available in fitted_components."""
        try:
            return all(
                step.name in self.fitted_components
                for step in self.pipeline_config.steps
            )
        except Exception:
            return False

    def _handle_fit(self, data_group):
        if not self.grouped_samples[data_group]:
            print(f"No data available for fitting from {data_group.name}")
            return

        group_data = np.array(self.grouped_samples[data_group]).T[
            None, :, :
        ]  # (n_times, n_channels) -> (1, n_channels, n_times)
        op_id = f"fit_{data_group.name}_{len(self.pending_operations)}"

        # Fit only the next unfitted stateful step in pipeline order
        future = self.executor.submit(
            self.pipeline_config.fit_next_stateful,
            group_data,
            self.fitted_components,
        )
        self.pending_operations[op_id] = future
        print(
            f"Submitted async fit operation {op_id} with"
            f"{len(group_data)} samples from {data_group.name}"
        )

    def _handle_partial_fit(self, data_group):
        """Toggle periodic partial-fit mode for the given logical group."""
        if self.partial_fit_active:
            self.partial_fit_active = False
            self._partial_fit_group = None
            print(f"Disabled periodic partial-fit mode for {data_group.name}")
            return
        self.partial_fit_active = True
        self._partial_fit_group = data_group
        if data_group not in self.partial_fit_last_index:
            self.partial_fit_last_index[data_group] = 0
        print(
            f"Enabled periodic partial-fit mode for "
            f"{data_group.name} with {self.partial_fit_window_size=}"
        )

    def _handle_predict(self, data_group: Optional[IntEnum]):
        # Toggle inference mode
        self.inference_active = not self.inference_active
        print(
            "Real-time inference mode "
            + ("enabled" if self.inference_active else "disabled")
        )

    def _handle_prediction_result(self, future: Future):
        """Callback for completed prediction futures."""
        result = future.result()
        if result is None:
            print(
                "Inference skipped for this window"
                "(incomplete pipeline or invalid data)"
            )
            return
        if self.queue:
            self.queue.put(asdict(result))

        print(
            f"Window {result.window_metadata.window_id}: prediction={result.prediction}, "
            f"probability={result.probability:.3f}, "
            f"time={result.processing_time:.3f}s"
        )

    def _maybe_process_latest_window(self):
        """Process latest window if we've accumulated enough new samples"""
        samples_since_last = (
            len(self.processed_samples) - self.last_processed_index
        )

        if samples_since_last >= self.step_size:
            window_data = np.array(
                # (n_times, n_channels) -> (1, n_channels, n_times)
                self.processed_samples[-self.window_size :]
            ).T[None, :, :]
            window_timestamps = self.samples_timestamps[-self.window_size :]

            window_metadata = WindowMetadata(
                start_timestamp=window_timestamps[0],
                end_timestamp=window_timestamps[-1],
                window_id=f"window_{self.window_id}",
            )

            future = self.executor.submit(
                process_window_in_pool,
                window_data,
                self.fitted_components.copy(),
                self.pipeline_config,
                window_metadata,
            )
            future.add_done_callback(self._handle_prediction_result)

            self.last_processed_index += self.step_size
            self.window_id += 1

    def _maybe_partial_fit_latest_window(self):
        """Submit non-overlapping partial-fit updates on the active group."""
        group = self._partial_fit_group
        if group is None:
            return
        samples_buffer = self.grouped_samples[group]
        start_index = self.partial_fit_last_index[group]

        available = len(samples_buffer) - start_index
        if available < self.partial_fit_window_size:
            return

        while (
            len(samples_buffer) - start_index >= self.partial_fit_window_size
        ):
            window_slice = samples_buffer[
                start_index : start_index + self.partial_fit_window_size
            ]
            window_data = np.array(window_slice).T[
                None, :, :
            ]  # (n_times, n_channels) -> (1, n_channels, n_times)

            op_id = f"partial_fit_window_{group.name}_{self.window_id}_{len(self.pending_operations)}"
            future = self.executor.submit(
                self.pipeline_config.partial_fit_components,
                window_data,
                self.fitted_components,
            )
            self.pending_operations[op_id] = future
            print(
                f"Submitted async partial-fit {op_id} using non-overlapping window starting at index {start_index}"
            )

            start_index += self.partial_fit_window_size

        self.partial_fit_last_index[group] = start_index


class EEGService:
    def __init__(
        self,
        pipeline_config,
        tcp_host: str | None = None,
        tcp_port: int | None = None,
        tcp_retries: int | None = None,
        state_config_path: Optional[str] = None,
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
                    self.state_manager.handle_events([e[0] for e in events])
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
        print("Stopping EEG service...")
        self.state_manager.inference_active = False
        self.tcp_sender.stop()
        self.executor.shutdown(wait=True)
        print("EEG service stopped")
