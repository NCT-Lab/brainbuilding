import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from enum import IntEnum

# Preprocessing configuration
ORDER = 4
LAG = 8
REMOVE_HEOG = True
REMOVE_VEOG = True

PICK_CHANNELS = np.array(
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

EVENT_TYPES = [
    "Arrow",
    "Background",
    "Cross",
    "Image",
    "Instruction",
    "Pause",
    "Rest",
    "VAS",
]

ND_CHANNELS = np.array(PICK_CHANNELS)
ND_CHANNELS_MASK = np.logical_or(ND_CHANNELS == "C3", ND_CHANNELS == "C4")

LOW_FREQ = 1.0
HIGH_FREQ = 40.0

# Online processing configuration
ONLINE_PROCESSING_LOW_FREQ = 4.0
ONLINE_PROCESSING_HIGH_FREQ = 40.0
ONLINE_PROCESSING_FILTER_ORDER = 4

# Stream configuration
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

REFERENCE_CHANNEL = "Fz"
REF_CHANNEL_IND = [
    i for i, ch in enumerate(CHANNELS_IN_STREAM) if ch == REFERENCE_CHANNEL
][0]
CHANNELS_MASK = np.array(
    [True if ch in CHANNELS_TO_KEEP else False for ch in CHANNELS_IN_STREAM]
)

# Default stream names
DEFAULT_EEG_STREAM_NAME = "NeoRec21-1247"
DEFAULT_EVENT_STREAM_NAME = "Brainbuilding-Events"
DEFAULT_EEG_STREAM_NAMES = ["NeoRec21-1247", "actiCHamp-23090108"]

# Processing parameters
DEFAULT_PROCESSING_WINDOW_SIZE = 250  # Points for classification
DEFAULT_PROCESSING_STEP = 125  # Points between classifications
DEFAULT_SFREQ = 250.0  # Sampling frequency
DEFAULT_SCALE_FACTOR = 1_000_000  # µV to V conversion

# TCP configuration
DEFAULT_TCP_HOST = "192.168.1.97"
DEFAULT_TCP_PORT = 8080
DEFAULT_TCP_RETRIES = 500

DEFAULT_HISTORY_PATH = "eeg_history.pickle"

# Debug settings
DEBUG = False


class Event(IntEnum):
    """Event IDs from Brainbuilding-Events stream"""

    BACKGROUND = 1
    REST = 2
    PREPARE = 3
    LEFT_HAND = 4
    RIGHT_HAND = 5

    BLINK = 10
    TRACK_DOT = 11
    RANDOM_DOT = 12

    @classmethod
    def is_calibration_event(cls, event) -> bool:
        return event in [cls.BLINK, cls.TRACK_DOT, cls.RANDOM_DOT]


# TODO: break this class fields into multiple config classes
@dataclass
class EEGProcessingConfig:
    """Configuration for EEG processing"""

    # Channel configuration
    channels: List[str] = None
    reference_channel: str = REFERENCE_CHANNEL
    channels_to_keep: List[str] = None

    # Processing parameters
    processing_window_size: int = DEFAULT_PROCESSING_WINDOW_SIZE
    processing_step: int = DEFAULT_PROCESSING_STEP
    sfreq: float = DEFAULT_SFREQ

    # Filter parameters
    low_freq: float = ONLINE_PROCESSING_LOW_FREQ
    high_freq: float = ONLINE_PROCESSING_HIGH_FREQ
    filter_order: int = ONLINE_PROCESSING_FILTER_ORDER

    # ICA parameters
    remove_veog: bool = REMOVE_VEOG
    remove_heog: bool = REMOVE_HEOG
    n_components: Optional[int] = None

    # Stream configuration
    eeg_stream_name: str = DEFAULT_EEG_STREAM_NAME
    event_stream_name: str = DEFAULT_EVENT_STREAM_NAME

    # TCP configuration
    tcp_host: str = DEFAULT_TCP_HOST
    tcp_port: int = DEFAULT_TCP_PORT
    tcp_retries: int = DEFAULT_TCP_RETRIES

    # File paths
    pipeline_path: str = ""
    history_path: str = DEFAULT_HISTORY_PATH

    # Misc
    scale_factor: float = DEFAULT_SCALE_FACTOR
    debug: bool = DEBUG

    def __post_init__(self):
        if self.channels is None:
            self.channels = list(CHANNELS_IN_STREAM)
        if self.channels_to_keep is None:
            self.channels_to_keep = list(CHANNELS_TO_KEEP)
        if self.n_components is None:
            self.n_components = len(self.channels_to_keep)

        # Calculate derived values
        self.ref_idx = self.channels.index(self.reference_channel)
        self.keep_channels = [
            i
            for i, ch in enumerate(self.channels)
            if ch != self.reference_channel
        ]
        self.channels_mask = np.array(
            [
                True if ch in self.channels_to_keep else False
                for ch in self.channels
            ]
        )


TRAINING_DATA_EVENT_NAMES = [
    "Animation@imagin/move/anim@left/hand",
    "Animation@imagin/move/anim@right/hand",
    "Animation@real/move/anim@left/hand",
    "Animation@real/move/anim@right/hand",
    "Arrow@imagin/move/anim@left/hand",
    "Arrow@imagin/move/anim@right/hand",
    "Arrow@real/move/anim@left/hand",
    "Arrow@real/move/anim@right/hand",
    "Background@@",
    "Cross@imagin/move/anim@left/hand",
    "Cross@imagin/move/anim@right/hand",
    "Cross@real/move/anim@left/hand",
    "Cross@real/move/anim@right/hand",
    "Instruction@@",
    "MFI@@",
    "Pause@@",
    "Rest@imagin/move/anim@left/hand",
    "Rest@imagin/move/anim@right/hand",
    "Rest@real/move/anim@left/hand",
    "Rest@real/move/anim@right/hand",
    "VAS@@",
]

VALIDATION_DATA_EVENT_NAMES = [
    "Animation@imagin/move/anim@left/hand",
    "Animation@imagin/move/anim@right/hand",
    "Background@@",
    "Cross@imagin/move/anim@left/hand",
    "Cross@imagin/move/anim@right/hand",
    "EyeWarmupBlink@@",
    "EyeWarmupMove@@",
    "EyeWarmupText@@",
    "Rest@imagin/move/anim@left/hand",
    "Rest@imagin/move/anim@right/hand",
]
ALL_EVENT_NAMES = list(
    set(TRAINING_DATA_EVENT_NAMES + VALIDATION_DATA_EVENT_NAMES)
)

CSP_METRIC = "riemann"

# Labels scheme: [is_imagery, is_background, is_righthand, is_point]
IMAGERY_CLASS_INDEX = 0
BACKGROUND_CLASS_INDEX = 1
RIGHT_HAND_CLASS_INDEX = 2
POINT_CLASS_INDEX = 3

LABEL_MAPPING = {
    "Point@@left/hand": [1, 0, 0, 1],
    "Point@@right/hand": [1, 0, 1, 1],
    "Image@@left/hand": [1, 0, 0, 0],
    "Image@@right/hand": [1, 0, 1, 0],
    "Rest@@left/hand": [0, 0, 0, 0],
    "Rest@@right/hand": [0, 0, 1, 0],
    "Background@@": [0, 1, 0, 0],
}


def STANDARD_EVENT_NAME_TO_ID_MAPPING(x):
    return LABEL_MAPPING.get(x, [])
