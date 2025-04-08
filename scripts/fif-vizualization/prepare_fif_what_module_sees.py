# %%

import numpy as np
import mne
from config import PICK_CHANNELS
import pickle

mne.set_config('MNE_BROWSER_BACKEND', 'qt')

with open('eeg_history.pickle', 'rb') as f:
    eeg_history = pickle.load(f)

processed_and_cleaned_data: list[tuple[np.ndarray, np.ndarray]] = eeg_history['processed_and_raw_windows']
processed_data = np.hstack([i[0] for i in processed_and_cleaned_data])
cleaned_data = np.hstack([i[1] for i in processed_and_cleaned_data])

# Create annotations for windows
n_windows = len(processed_and_cleaned_data)
window_size = processed_and_cleaned_data[0][0].shape[1]  # n_times for each window
sfreq = 500  # Your sampling frequency

# Calculate onset times in seconds for each window
onset_times = np.arange(n_windows) * window_size / sfreq
durations = [window_size / sfreq] * n_windows  # Duration of each window in seconds
descriptions = [f'Window_{i}' for i in range(n_windows)]

# Create annotations object
annotations = mne.Annotations(
    onset=onset_times,
    duration=durations,
    description=descriptions
)

RAW_PICK_CHANNELS = [f'{i}-raw' for i in PICK_CHANNELS]
info_raw = mne.create_info(ch_names=RAW_PICK_CHANNELS, sfreq=sfreq, ch_types='eeg')
raw_processed = mne.io.RawArray(processed_data, info_raw)
info_cleaned = mne.create_info(ch_names=PICK_CHANNELS.tolist(), sfreq=sfreq, ch_types='eeg')
raw_cleaned = mne.io.RawArray(cleaned_data, info_cleaned)
raw = raw_processed.add_channels([raw_cleaned], force_update_info=True)
pick_and_raw = [i for pair in [[i, f'{i}-raw'] for i in PICK_CHANNELS] for i in pair]
raw.pick(picks=pick_and_raw)

# Set the annotations to the raw object
raw.set_annotations(annotations)

# Save the raw data to a FIFF file
raw.save('eeg_data_raw.fif', overwrite=True)

raw.plot(block=True)
# %%

# raw.drop_channels(["Fp1-1"])
# raw.set_eeg_reference(ref_channels=['Fz'], verbose='error')
# raw.rename_channels({"Fp1-0": "Fp1"})
# raw = raw.pick(picks=PICK_CHANNELS)
# sfreq = raw.info["sfreq"]
# raw._data = apply_bandpass_filter(raw.get_data(), sfreq)

# raw_old = raw.copy()
# raw_old.rename_channels({k: f'{k}-raw' for k in raw.ch_names})

# raw = apply_ica(raw)
# raw = raw.pick(picks=PICK_CHANNELS)
# raw.add_channels([raw_old], force_update_info=True)
# pick_and_raw = [i for pair in [[i, f'{i}-raw'] for i in PICK_CHANNELS] for i in pair]
# raw.pick(picks=pick_and_raw)
# raw.plot(block=False)
