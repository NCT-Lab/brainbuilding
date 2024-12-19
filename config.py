import numpy as np

ORDER = 4
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


non_class_filter = lambda x: (
    'Rest' not in x 
    and 'Animation' not in x 
    and 'imagin' not in x 
    and 'real' not in x
)

zero_filter = lambda x: 'Rest' in x
one_filter = lambda x: 'Animation' in x

STANDARD_EVENT_NAME_TO_ID_MAPPING = {
    i: 100 if non_class_filter(i) else (
        0 if zero_filter(i) else (
            1 if one_filter(i) else 100
        )
    )
    for i in ALL_EVENT_NAMES
}

# STANDARD_EVENT_NAME_TO_ID_MAPPING = {
#     'Rest@imagin/move/anim@left/hand': 2,
#     'Rest@imagin/move/anim@right/hand': 3,
#     'Animation@imagin/move/anim@left/hand': 4,
#     'Animation@imagin/move/anim@right/hand': 5,
#     'Rest@real/move/anim@left/hand': 6,
#     'Rest@real/move/anim@right/hand': 7,
#     'Animation@real/move/anim@left/hand': 8,
#     'Animation@real/move/anim@right/hand': 9,
# }
# STANDARD_EVENT_NAME_TO_ID_MAPPING_EXT = {
#     i: 100 
#     for i in ALL_EVENT_NAMES
#     if i not in STANDARD_EVENT_NAME_TO_ID_MAPPING
# }
# STANDARD_EVENT_NAME_TO_ID_MAPPING.update(STANDARD_EVENT_NAME_TO_ID_MAPPING_EXT)

