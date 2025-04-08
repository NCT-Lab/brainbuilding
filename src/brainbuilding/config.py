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

LOW_FREQ = 4.
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

# non_class_filter = lambda x: (
#     'Rest' not in x 
#     and 'Animation' not in x 
#     and 'Background' not in x
# )
ONE_CLASS_EVENT_NAMES = [
    "Point@@left/hand",
    "Point@@right/hand",
    "Image@@left/hand",
    "Image@@right/hand",
]
ZERO_CLASS_EVENT_NAMES = [
    "Rest@@left/hand",
    "Rest@@right/hand",
]

def is_event_name_zero_class(event_name):
    return any([i in event_name for i in ZERO_CLASS_EVENT_NAMES])

def is_event_name_one_class(event_name):
    return any([i in event_name for i in ONE_CLASS_EVENT_NAMES])

STANDARD_EVENT_NAME_TO_ID_MAPPING = lambda x: 0 if is_event_name_zero_class(x) else (
    1 if is_event_name_one_class(x) else 2
)


