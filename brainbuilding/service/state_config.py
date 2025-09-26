from __future__ import annotations

from typing import Dict, List, Optional, Type, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field, PrivateAttr, model_validator  # type: ignore  # noqa: E0401
from enum import IntEnum
import yaml  # type: ignore

from brainbuilding.service.state_types import (
    StateDefinition,
    TransitionAction,
    ACTIONS_REQUIRING_GROUP,
)


@dataclass
class StateMachineRuntime:
    """Structured result of building the runtime state machine."""

    processing_state_enum: Type[IntEnum]
    logical_group_enum: Type[IntEnum]
    states: Dict[IntEnum, StateDefinition]
    window_size: int
    step_size_by_action: Dict[TransitionAction, int]


class StateConfig(BaseModel):
    """Declarative configuration for a single state.

    All enum-typed fields are represented by their names in YAML and are
    validated into the corresponding enums by Pydantic.
    """

    accepted_events: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of event names to next state names",
    )
    # Use string names in YAML, convert to enums later
    data_collection_group: Optional[str] = None
    on_entry_actions: Optional[List[Union["ActionSpecModel", str]]] = None
    on_exit_actions: Optional[List[Union["ActionSpecModel", str]]] = None
    on_transition_actions: Optional[
        Dict[str, List[Union["ActionSpecModel", str]]]
    ] = None
    # Trigger-specific action->group mappings
    class_label: int = 0


class StateMachineConfig(BaseModel):
    """Top-level state machine configuration.

    - event_ids: mapping of event-name -> integer id received from the stream
    - states: mapping of state-name (ProcessingState.name) -> StateConfig
    """

    # Dynamic enums (names -> integer values)
    processing_states: Dict[str, int] = Field(default_factory=dict)
    logical_groups: Dict[str, int] = Field(default_factory=dict)

    # Stream event identifiers
    event_ids: Dict[str, int] = Field(default_factory=dict)
    states: Dict[str, StateConfig] = Field(default_factory=dict)
    # Windowing configuration (required)
    windowing: "WindowingConfig"

    # Private, strictly-typed, derived fields (built by validator)
    _proc_enum: Type[IntEnum] = PrivateAttr()
    _group_enum: Type[IntEnum] = PrivateAttr()
    _window_size: int = PrivateAttr(default=250)
    _step_by_action: Dict[TransitionAction, int] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _finalize(self) -> "StateMachineConfig":
        self._proc_enum, self._group_enum = _build_enums(self)
        self._window_size = self.windowing.window_size
        self._step_by_action = {
            action: self._window_size for action in TransitionAction
        }
        for action_name, step_val in self.windowing.step_sizes.items():
            if action_name not in TransitionAction.__members__:
                raise ValueError(
                    f"Unknown action in step_sizes: {action_name}"
                )
            self._step_by_action[TransitionAction[action_name]] = int(step_val)
        return self

    # TODO: надо все это дело нормально переписать без этой императивщины
    @property
    def runtime(self) -> StateMachineRuntime:
        # Precompute name->enum map once
        state_name_to_enum: Dict[str, IntEnum] = {
            n: self._proc_enum[n] for n in self.states
        }
        # States (pure comprehensions)
        states_rt: Dict[IntEnum, StateDefinition] = {}
        for state_name, sc in self.states.items():
            state_enum = state_name_to_enum[state_name]
            acc_events: Dict[int, IntEnum] = {
                self.event_ids[event]: state_name_to_enum[next_state]
                for event, next_state in sc.accepted_events.items()
            }
            # Inline action specs: build actions and delay mappings
            entry_actions, entry_delays, entry_groups = _map_action_specs_list(
                sc.on_entry_actions
            )
            exit_actions, exit_delays, exit_groups = _map_action_specs_list(
                sc.on_exit_actions
            )
            on_trans: Dict[IntEnum, List[TransitionAction]] = {}
            on_trans_delays: Dict[IntEnum, Dict[TransitionAction, float]] = {}
            on_trans_groups: Dict[
                IntEnum, Dict[TransitionAction, IntEnum]
            ] = {}
            for n, a in (sc.on_transition_actions or {}).items():
                next_enum = state_name_to_enum[n]
                acts, delays, groups = _map_action_specs_list(a)
                on_trans[next_enum] = acts
                on_trans_delays[next_enum] = delays
                on_trans_groups[next_enum] = _map_string_groups_to_enum(
                    groups, self._group_enum
                )

            states_rt[state_enum] = StateDefinition(
                name=state_enum,
                accepted_events=acc_events,
                class_label=sc.class_label,
                data_collection_group=(
                    self._group_enum[sc.data_collection_group]
                    if isinstance(sc.data_collection_group, str)
                    else sc.data_collection_group
                )
                if sc.data_collection_group is not None
                else None,
                on_entry_actions=entry_actions,
                on_exit_actions=exit_actions,
                on_transition_actions=on_trans,
                on_entry_action_groups=_map_string_groups_to_enum(
                    entry_groups, self._group_enum
                ),
                on_exit_action_groups=_map_string_groups_to_enum(
                    exit_groups, self._group_enum
                ),
                on_transition_action_groups=on_trans_groups,
                on_entry_action_delays=entry_delays,
                on_exit_action_delays=exit_delays,
                on_transition_action_delays=on_trans_delays,
            )

        # Validation step
        for state_enum, state_def in states_rt.items():
            # Check on_entry actions
            if state_def.on_entry_actions:
                for action in state_def.on_entry_actions:
                    if action in ACTIONS_REQUIRING_GROUP:
                        group = (
                            state_def.on_entry_action_groups.get(action)
                            or state_def.data_collection_group
                        )
                        if group is None:
                            raise ValueError(
                                f"State '{state_enum.name}': on_entry action '{action.name}' requires a group, but none is defined."
                            )

            # Check on_exit actions
            if state_def.on_exit_actions:
                for action in state_def.on_exit_actions:
                    if action in ACTIONS_REQUIRING_GROUP:
                        group = (
                            state_def.on_exit_action_groups.get(action)
                            or state_def.data_collection_group
                        )
                        if group is None:
                            raise ValueError(
                                f"State '{state_enum.name}': on_exit action '{action.name}' requires a group, but none is defined."
                            )

            # Check on_transition actions
            if state_def.on_transition_actions:
                for (
                    next_state_enum,
                    actions,
                ) in state_def.on_transition_actions.items():
                    for action in actions:
                        if action in ACTIONS_REQUIRING_GROUP:
                            transition_groups = (
                                state_def.on_transition_action_groups.get(
                                    next_state_enum, {}
                                )
                            )
                            group = (
                                transition_groups.get(action)
                                or state_def.data_collection_group
                            )
                            if group is None:
                                raise ValueError(
                                    f"State '{state_enum.name}': transition to '{next_state_enum.name}' action '{action.name}' requires a group, but none is defined."
                                )
        return StateMachineRuntime(
            processing_state_enum=self._proc_enum,
            logical_group_enum=self._group_enum,
            states=states_rt,
            window_size=self._window_size,
            step_size_by_action=self._step_by_action,
        )


class WindowingConfig(BaseModel):
    window_size: int = Field(
        ..., description="Global window size for all actions"
    )
    # Mapping of TransitionAction names -> step size integers
    step_sizes: Dict[str, int] = Field(default_factory=dict)


# -----------------------------
# Runtime builder (strict, typed)
# -----------------------------


def _build_enums(
    cfg: StateMachineConfig,
) -> tuple[Type[IntEnum], Type[IntEnum]]:
    proc_enum: Type[IntEnum] = IntEnum(  # type: ignore[misc]
        "ProcessingState", cfg.processing_states or {}
    )
    group_enum: Type[IntEnum] = IntEnum(  # type: ignore[misc]
        "LogicalGroup", cfg.logical_groups or {}
    )
    return proc_enum, group_enum


def _map_actions_list(actions: Optional[List[str]]) -> List[TransitionAction]:
    if not actions:
        return []
    return [
        TransitionAction[a] if isinstance(a, str) else TransitionAction(a)
        for a in actions
    ]


def _map_action_groups_names(
    mapping: Optional[Dict[str, str]],
    group_enum: Type[IntEnum],
) -> Dict[TransitionAction, IntEnum]:
    if not mapping:
        return {}
    result: Dict[TransitionAction, IntEnum] = {}
    for action_name, group_name in mapping.items():
        result[TransitionAction[action_name]] = group_enum[group_name]
    return result


class ActionSpecModel(BaseModel):
    name: str
    delay_seconds: float = Field(default=0.0, ge=0.0)
    group: Optional[str] = None


def _map_action_specs_list(
    actions: Optional[List[Union[ActionSpecModel, str]]],
) -> tuple[
    List[TransitionAction],
    Dict[TransitionAction, float],
    Dict[TransitionAction, str],
]:
    if not actions:
        return [], {}, {}
    action_list: List[TransitionAction] = []
    delays: Dict[TransitionAction, float] = {}
    groups: Dict[TransitionAction, str] = {}
    for item in actions:
        if isinstance(item, str):
            if item not in TransitionAction.__members__:
                raise ValueError(f"Unknown action: {item}")
            action = TransitionAction[item]
            delay = 0.0
            group = None
        else:
            if item.name not in TransitionAction.__members__:
                raise ValueError(f"Unknown action: {item.name}")
            action = TransitionAction[item.name]
            delay = float(item.delay_seconds)
            group = item.group
        action_list.append(action)
        if delay > 0.0:
            delays[action] = delay
        if group is not None:
            groups[action] = group
    return action_list, delays, groups


def load_state_machine_from_yaml(path: str) -> StateMachineRuntime:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = StateMachineConfig(**data)
    return cfg.runtime


def _map_string_groups_to_enum(
    mapping: Dict[TransitionAction, str],
    group_enum: Type[IntEnum],
) -> Dict[TransitionAction, IntEnum]:
    return {
        action: group_enum[group_name]
        for action, group_name in mapping.items()
    }
