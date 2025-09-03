from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass

from pydantic import BaseModel, Field  # type: ignore
from enum import IntEnum
import yaml  # type: ignore

from brainbuilding.service.state_types import StateDefinition, TransitionAction


@dataclass
class StateMachineRuntime:
    """Structured result of building the runtime state machine."""

    processing_state_enum: Type[IntEnum]
    logical_group_enum: Type[IntEnum]
    states: Dict[IntEnum, StateDefinition]
    metadata: Dict[str, Any]


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
    on_entry_actions: Optional[List[str]] = None
    on_exit_actions: Optional[List[str]] = None
    on_transition_actions: Optional[Dict[str, List[str]]] = None
    # Trigger-specific action->group mappings
    on_entry_action_groups: Optional[Dict[str, str]] = None
    on_exit_action_groups: Optional[Dict[str, str]] = None
    on_transition_action_groups: Optional[Dict[str, Dict[str, str]]] = None


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
    # Reserved for future metadata; keep empty for now

    def build_runtime_states(self) -> StateMachineRuntime:
        """Build a fully validated runtime representation (structured)."""
        proc_enum, group_enum = self._build_enums()
        state_name_to_enum = {s.name: s for s in proc_enum}
        self._validate_states(state_name_to_enum)
        self._validate_events()
        states = self._build_state_definitions(
            proc_enum, group_enum, state_name_to_enum
        )
        metadata = self._build_metadata(group_enum)
        return StateMachineRuntime(
            processing_state_enum=proc_enum,
            logical_group_enum=group_enum,
            states=states,
            metadata=metadata,
        )

    def _build_enums(self) -> Tuple[Type[IntEnum], Type[IntEnum]]:
        processing: Dict[str, int] = self.processing_states or {}
        groups: Dict[str, int] = self.logical_groups or {}
        proc_enum: Type[IntEnum] = IntEnum(  # type: ignore[misc]
            "ProcessingState",
            processing,
        )
        group_enum: Type[IntEnum] = IntEnum(  # type: ignore[misc]
            "LogicalGroup",
            groups,
        )
        return proc_enum, group_enum

    def _validate_states(self, state_name_to_enum: Dict[str, IntEnum]) -> None:
        for state_name in self.states.keys():
            if state_name not in state_name_to_enum:
                raise ValueError(f"Unknown state name in config: {state_name}")

    def _validate_events(self) -> None:
        for event_name, event_id in self.event_ids.items():
            if not isinstance(event_id, int):
                raise ValueError(
                    f"Event id for {event_name} must be int, got {type(event_id)}"
                )

    def _build_state_definitions(
        self,
        proc_enum: Type[IntEnum],
        group_enum: Type[IntEnum],
        state_name_to_enum: Dict[str, IntEnum],
    ) -> Dict[IntEnum, StateDefinition]:
        runtime_states: Dict[IntEnum, StateDefinition] = {}
        for state_name, sc in self.states.items():
            state_enum = state_name_to_enum[state_name]
            accepted_events = self._map_accepted_events(
                sc, state_name, state_name_to_enum
            )
            on_transition_actions = self._map_on_transition_actions(
                sc, state_name_to_enum
            )
            on_entry_action_groups = self._map_action_groups(
                sc.on_entry_action_groups, group_enum
            )
            on_exit_action_groups = self._map_action_groups(
                sc.on_exit_action_groups, group_enum
            )
            on_transition_action_groups = (
                self._map_on_transition_action_groups(
                    sc, state_name_to_enum, group_enum
                )
            )

            runtime_states[state_enum] = StateDefinition(
                name=state_enum,
                accepted_events=accepted_events,
                data_collection_group=(
                    group_enum[sc.data_collection_group]
                    if isinstance(sc.data_collection_group, str)
                    else sc.data_collection_group
                )
                if sc.data_collection_group is not None
                else None,
                on_entry_actions=self._map_actions_list(sc.on_entry_actions),
                on_exit_actions=self._map_actions_list(sc.on_exit_actions),
                on_transition_actions=on_transition_actions,
                on_entry_action_groups=on_entry_action_groups,
                on_exit_action_groups=on_exit_action_groups,
                on_transition_action_groups=on_transition_action_groups,
            )
        return runtime_states

    def _map_accepted_events(
        self,
        sc: StateConfig,
        state_name: str,
        state_name_to_enum: Dict[str, IntEnum],
    ) -> Dict[int, IntEnum]:
        mapped: Dict[int, IntEnum] = {}
        for event_name, next_state_name in sc.accepted_events.items():
            if event_name not in self.event_ids:
                raise ValueError(
                    f"Event '{event_name}' used in state '{state_name}' is not defined in event_ids"
                )
            if next_state_name not in state_name_to_enum:
                raise ValueError(
                    (
                        f"Next state '{next_state_name}' in state "
                        f"'{state_name}' "
                        "is not a valid ProcessingState"
                    )
                )
            mapped[self.event_ids[event_name]] = state_name_to_enum[
                next_state_name
            ]
        return mapped

    def _map_on_transition_actions(
        self,
        sc: StateConfig,
        state_name_to_enum: Dict[str, IntEnum],
    ) -> Dict[IntEnum, List[TransitionAction]]:
        if not sc.on_transition_actions:
            return {}
        converted: Dict[IntEnum, List[TransitionAction]] = {}
        for next_state_name, actions in sc.on_transition_actions.items():
            if next_state_name not in state_name_to_enum:
                raise ValueError(
                    f"on_transition_actions references unknown state '{next_state_name}'"
                )
            converted[state_name_to_enum[next_state_name]] = self._map_actions_list(
                actions
            )
        return converted

    def _map_actions_list(self, actions: Optional[List[str]]) -> List[TransitionAction]:
        if not actions:
            return []
        return [
            TransitionAction[a] if isinstance(a, str) else TransitionAction(a)
            for a in actions
        ]

    def _map_action_groups(
        self, mapping: Optional[Dict[str, str]], group_enum: Type[IntEnum]
    ) -> Dict[TransitionAction, IntEnum]:
        if not mapping:
            return {}
        result: Dict[TransitionAction, IntEnum] = {}
        for action_name, group_name in mapping.items():
            action = (
                TransitionAction[action_name]
                if isinstance(action_name, str)
                else TransitionAction(action_name)
            )
            group = (
                group_enum[group_name]
                if isinstance(group_name, str)
                else group_name
            )
            result[action] = group
        return result

    def _map_on_transition_action_groups(
        self,
        sc: StateConfig,
        state_name_to_enum: Dict[str, IntEnum],
        group_enum: Type[IntEnum],
    ) -> Dict[IntEnum, Dict[TransitionAction, IntEnum]]:
        if not sc.on_transition_action_groups:
            return {}
        result: Dict[IntEnum, Dict[TransitionAction, IntEnum]] = {}
        for next_state_name, mapping in sc.on_transition_action_groups.items():
            if next_state_name not in state_name_to_enum:
                raise ValueError(
                    f"on_transition_action_groups references unknown state '{next_state_name}'"
                )
            next_enum = state_name_to_enum[next_state_name]
            inner: Dict[TransitionAction, IntEnum] = {}
            for action_name, group_name in mapping.items():
                action = (
                    TransitionAction[action_name]
                    if isinstance(action_name, str)
                    else TransitionAction(action_name)
                )
                group = (
                    group_enum[group_name]
                    if isinstance(group_name, str)
                    else group_name
                )
                inner[action] = group
            result[next_enum] = inner
        return result

    def _build_metadata(self, group_enum: Type[IntEnum]) -> Dict[str, Any]:
        # No dynamic metadata at the moment
        return {}


def load_state_machine_from_yaml(path: str) -> StateMachineRuntime:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cfg = StateMachineConfig(**data)
    return cfg.build_runtime_states()
