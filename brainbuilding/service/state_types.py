from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class TransitionAction(IntEnum):
    DO_NOTHING = 0
    FIT = 1
    PARTIAL_FIT = 2
    PREDICT = 3
    COLLECT_FOR_TRAIN = 4


ACTIONS_REQUIRING_GROUP = {
    TransitionAction.FIT,
    TransitionAction.PARTIAL_FIT,
    TransitionAction.PREDICT,
    TransitionAction.COLLECT_FOR_TRAIN,
}


@dataclass
class StateDefinition:
    name: IntEnum
    accepted_events: Dict[int, IntEnum]
    on_entry_actions: Optional[List[TransitionAction]]
    on_exit_actions: Optional[List[TransitionAction]]
    on_transition_actions: Optional[Dict[IntEnum, List[TransitionAction]]]
    class_label: int = 0
    data_collection_group: Optional[IntEnum] = None
    on_entry_action_groups: Dict[TransitionAction, IntEnum] = field(
        default_factory=dict
    )
    on_exit_action_groups: Dict[TransitionAction, IntEnum] = field(
        default_factory=dict
    )
    on_transition_action_groups: Dict[
        IntEnum, Dict[TransitionAction, IntEnum]
    ] = field(default_factory=dict)
    # Per-trigger action delays (seconds)
    on_entry_action_delays: Dict[TransitionAction, float] = field(
        default_factory=dict
    )
    on_exit_action_delays: Dict[TransitionAction, float] = field(
        default_factory=dict
    )
    on_transition_action_delays: Dict[
        IntEnum, Dict[TransitionAction, float]
    ] = field(default_factory=dict)
