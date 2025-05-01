import numpy as np

ORDER = 4
LAG = 8
REMOVE_HEOG = True
REMOVE_VEOG = True

PICK_CHANNELS = np.array([
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Cz", "Pz"
])

EVENT_TYPES = [
    'Arrow', 'Background', 'Cross', 'Image', 'Instruction',
    'Pause', 'Rest', 'VAS'
]

ND_CHANNELS = np.array(PICK_CHANNELS)
ND_CHANNELS_MASK = np.logical_or(ND_CHANNELS == 'C3', ND_CHANNELS == 'C4')

LOW_FREQ = 1.
HIGH_FREQ = 40.

TRAINING_DATA_EVENT_NAMES = [
    'Animation@imagin/move/anim@left/hand',
    'Animation@imagin/move/anim@right/hand',
    'Animation@real/move/anim@left/hand',
    'Animation@real/move/anim@right/hand',
    'Arrow@imagin/move/anim@left/hand',
    'Arrow@imagin/move/anim@right/hand',
    'Arrow@real/move/anim@left/hand',
    'Arrow@real/move/anim@right/hand',
    'Background@@',
    'Cross@imagin/move/anim@left/hand',
    'Cross@imagin/move/anim@right/hand',
    'Cross@real/move/anim@left/hand',
    'Cross@real/move/anim@right/hand',
    'Instruction@@',
    'MFI@@',
    'Pause@@',
    'Rest@imagin/move/anim@left/hand',
    'Rest@imagin/move/anim@right/hand',
    'Rest@real/move/anim@left/hand',
    'Rest@real/move/anim@right/hand',
    'VAS@@'
]

VALIDATION_DATA_EVENT_NAMES = [
    'Animation@imagin/move/anim@left/hand',
    'Animation@imagin/move/anim@right/hand',
    'Background@@',
    'Cross@imagin/move/anim@left/hand',
    'Cross@imagin/move/anim@right/hand',
    'EyeWarmupBlink@@',
    'EyeWarmupMove@@',
    'EyeWarmupText@@',
    'Rest@imagin/move/anim@left/hand',
    'Rest@imagin/move/anim@right/hand'
]
ALL_EVENT_NAMES = list(set(TRAINING_DATA_EVENT_NAMES + VALIDATION_DATA_EVENT_NAMES))

CSP_METRIC = 'riemann'

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
# ONE_CLASS_EVENT_NAMES = [
#     "Point@@left/hand",
#     "Point@@right/hand",
#     "Image@@left/hand",
#     "Image@@right/hand",
# ]
# ZERO_CLASS_EVENT_NAMES = [
#     "Rest@@left/hand",
#     "Rest@@right/hand",
# ]
# BACKGROUND_EVENT_NAMES = [
#     "Background@@",
# ]

def is_event_name_zero_class(event_name):
    return any([i in event_name for i in ZERO_CLASS_EVENT_NAMES])

def is_event_name_one_class(event_name):
    return any([i in event_name for i in ONE_CLASS_EVENT_NAMES])

def is_event_name_background(event_name):
    return any([i in event_name for i in BACKGROUND_EVENT_NAMES])

def STANDARD_EVENT_NAME_TO_ID_MAPPING(x):
    return LABEL_MAPPING.get(x, [])
    # if is_event_name_zero_class(x):
    #     return 0
    # if is_event_name_one_class(x):
    #     return 1
    # if is_event_name_background(x):
    #     return 2
    # return -1


