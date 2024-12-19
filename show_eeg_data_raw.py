# %%
import mne

# Load the saved FIFF file
raw = mne.io.read_raw_fif('eeg_data_raw.fif', preload=True)

# Configure MNE to use Qt backend for plotting
mne.set_config('MNE_BROWSER_BACKEND', 'qt')

# Plot the loaded data
raw.plot(block=True)

# Print some basic information about the loaded data
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Number of time points: {len(raw.times)}")
print(f"Duration: {raw.times[-1]:.2f} seconds")
print(f"Number of annotations: {len(raw.annotations)}")
# %%
