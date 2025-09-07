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


@dataclass
class StateDefinition:
    name: IntEnum
    accepted_events: Dict[int, IntEnum]
    class_label: int = 0
    data_collection_group: Optional[IntEnum] = None
    on_entry_actions: List[TransitionAction] = field(default_factory=list)
    on_exit_actions: List[TransitionAction] = field(default_factory=list)
    on_transition_actions: Dict[IntEnum, List[TransitionAction]] = field(
        default_factory=dict
    )
    on_entry_action_groups: Dict[TransitionAction, IntEnum] = field(
        default_factory=dict
    )
    on_exit_action_groups: Dict[TransitionAction, IntEnum] = field(
        default_factory=dict
    )
    on_transition_action_groups: Dict[IntEnum, Dict[TransitionAction, IntEnum]] = field(
        default_factory=dict
    )
