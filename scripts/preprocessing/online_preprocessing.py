"""
Online-style preprocessing that mimics the real-time pipeline from main.py
but processes offline data to test equivalence with standard preprocessing.
"""
import warnings
import pyneurostim as ns
import numpy as np
import mne
from scipy.signal import butter, sosfilt, sosfilt_zi
import scipy.signal
import sys
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from brainbuilding.core.transformers import EyeRemoval
from src.brainbuilding.eye_removal import create_standard_eog_channels
from brainbuilding.core.config import PICK_CHANNELS, ND_CHANNELS_MASK, ORDER, REMOVE_HEOG, REMOVE_VEOG, LOW_FREQ, HIGH_FREQ

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", 'data/raw-dataset-2/')
TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", 'data/fif-dataset-online-preprocessed')

warnings.filterwarnings('ignore')
DOWNSAMPLE_SFREQ = 250 

# Online processing constants matching main.py
CHANNELS_IN_STREAM = np.array(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz'])
REF_CHANNEL_IND = [i for i, ch in enumerate(CHANNELS_IN_STREAM) if ch == 'Fz'][0]
CHANNELS_TO_KEEP = np.array([
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Cz", "Pz"
])
CHANNELS_MASK = np.array([True if ch in CHANNELS_TO_KEEP else False for ch in CHANNELS_IN_STREAM])

class OnlineSignalFilter:
    """Stateful bandpass filter for EEG data that mimics online processing"""
    def __init__(self, sfreq: float, n_channels: int):
        self.sfreq = sfreq
        self.sos = self._create_bandpass_filter(sfreq)
        # Initialize filter state for each channel
        self.zi = np.array([sosfilt_zi(self.sos) for _ in range(n_channels)])
        self.zi = np.rollaxis(self.zi, 0, 3)
        
    def _create_bandpass_filter(self, sfreq: float):
        """Create bandpass filter coefficients matching main.py"""
        nyq = sfreq / 2
        low = 4.0 / nyq  # Hardcoded in main.py
        high = 40.0 / nyq
        return butter(4, [low, high], btype='band', output='sos')
        
    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process single sample maintaining filter state"""
        sample = sample.reshape(1, -1)  # Ensure correct shape
        filtered, self.zi = sosfilt(self.sos, sample, axis=0, zi=self.zi)
        return filtered.flatten()

def create_standard_eog_channels_online(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create vEOG and hEOG channels from EEG data (matching main.py)"""
    # Get indices for EOG computation
    fp1 = data[CHANNELS_TO_KEEP == 'Fp1'][0]
    fp2 = data[CHANNELS_TO_KEEP == 'Fp2'][0]
    f7 = data[CHANNELS_TO_KEEP == 'F7'][0]
    f8 = data[CHANNELS_TO_KEEP == 'F8'][0]
    
    # Compute EOG channels
    veog = (fp2 + fp1) / 2
    heog = f8 - f7
    
    return veog, heog

def collect_warmup_data(raw: mne.io.Raw) -> Tuple[List[np.ndarray], List[float]]:
    """Collect data during EyeWarmupBlink@@ and EyeWarmupMove@@ events for ICA fitting"""
    warmup_data = []
    warmup_timestamps = []
    
    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Find warmup events
    warmup_event_names = ['EyeWarmupBlink@@', 'EyeWarmupMove@@']
    warmup_event_ids = []
    
    for name in warmup_event_names:
        if name in event_ids:
            warmup_event_ids.append(event_ids[name])
    
    if not warmup_event_ids:
        print("Warning: No warmup events found, using first 30 seconds of data")
        # Use first 30 seconds if no warmup events
        end_sample = min(int(30 * sfreq), data.shape[1])
        warmup_samples = data[:, :end_sample]
        for i in range(warmup_samples.shape[1]):
            warmup_data.append(warmup_samples[:, i])
            warmup_timestamps.append(i / sfreq)
        return warmup_data, warmup_timestamps
    
    # Collect data during warmup events
    for event in events:
        event_time, _, event_id = event
        if event_id in warmup_event_ids:
            # Find end of this event (next event or end of data)
            next_events = events[events[:, 0] > event_time]
            if len(next_events) > 0:
                end_time = next_events[0, 0]
            else:
                end_time = data.shape[1]
            
            # Collect samples during this event
            for sample_idx in range(event_time, min(end_time, data.shape[1])):
                warmup_data.append(data[:, sample_idx])
                warmup_timestamps.append(sample_idx / sfreq)
    
    print(f"Collected {len(warmup_data)} warmup samples")
    return warmup_data, warmup_timestamps

def online_preprocess_sample(sample: np.ndarray) -> np.ndarray:
    """Process a single sample exactly like the online pipeline"""
    # Sample comes in as (n_channels,) from raw data
    # Subtract reference channel (Fz)
    referenced = sample - sample[REF_CHANNEL_IND]
    
    # Convert from µV to V (assuming input is in µV)
    scaled = referenced / 1_000_000
    
    # Keep only needed channels
    selected = scaled[CHANNELS_MASK]
    
    return selected

def apply_online_preprocessing(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply online-style preprocessing to raw data"""
    print("Applying online-style preprocessing...")
    
    # Get original data and info
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Step 1: Collect warmup data for ICA fitting
    print("Collecting warmup data for ICA fitting...")
    warmup_data, warmup_timestamps = collect_warmup_data(raw)
    
    # Step 2: Process warmup data samples individually to build processed dataset
    processed_warmup_data = []
    for sample in warmup_data:
        processed_sample = online_preprocess_sample(sample)
        processed_warmup_data.append(processed_sample)
    
    processed_warmup_array = np.array(processed_warmup_data).T  # Shape: (n_channels, n_samples)
    
    # Step 3: Create EOG channels and fit ICA
    print("Fitting ICA on warmup data...")
    veog, heog = create_standard_eog_channels_online(processed_warmup_array)
    
    # Fit ICA using EyeRemoval (matching main.py implementation)
    ica = EyeRemoval(
        n_components=len(CHANNELS_TO_KEEP), 
        veog_data=veog, 
        heog_data=heog,
        remove_veog=REMOVE_VEOG,
        remove_heog=REMOVE_HEOG
    )
    ica.fit(processed_warmup_array)
    
    # Step 4: Initialize online filter
    signal_filter = OnlineSignalFilter(sfreq=sfreq, n_channels=len(CHANNELS_TO_KEEP))
    
    # Step 5: Process all data sample by sample
    print("Processing all samples with online pipeline...")
    processed_data = []
    
    for i in range(data.shape[1]):
        # Process sample (reference subtraction, scaling, channel selection)
        processed_sample = online_preprocess_sample(data[:, i])
        
        # Apply bandpass filter (stateful)
        filtered_sample = signal_filter.process_sample(processed_sample)
        
        processed_data.append(filtered_sample)
    
    # Convert to array
    processed_array = np.array(processed_data).T  # Shape: (n_channels, n_samples)
    
    # Step 6: Apply ICA to remove eye artifacts
    print("Applying ICA for eye artifact removal...")
    cleaned_data = ica.transform(processed_array)
    
    # Step 7: Create new Raw object with processed data
    # Create info for the new raw object
    new_info = mne.create_info(
        ch_names=list(CHANNELS_TO_KEEP),
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Create new raw object
    processed_raw = mne.io.RawArray(cleaned_data, new_info)
    
    # Copy annotations from original raw
    processed_raw.set_annotations(raw.annotations)
    
    return processed_raw

def find_nan_chunks(data):
    """Find consecutive chunks of NaN values in the signal."""
    nan_mask = np.any(np.isnan(data), axis=0)
    change_points = np.diff(nan_mask.astype(int))
    chunk_starts = np.where(change_points == 1)[0] + 1
    chunk_ends = np.where(change_points == -1)[0] + 1
    
    if nan_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if nan_mask[-1]:
        chunk_ends = np.append(chunk_ends, len(nan_mask))
        
    return list(zip(chunk_starts, chunk_ends))

def find_non_nan_chunks(data):
    """Find consecutive chunks of non-NaN values in the signal."""
    non_nan_mask = np.all(~np.isnan(data), axis=0)
    change_points = np.diff(non_nan_mask.astype(int))
    chunk_starts = np.where(change_points == 1)[0] + 1
    chunk_ends = np.where(change_points == -1)[0] + 1
    
    if non_nan_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if non_nan_mask[-1]:
        chunk_ends = np.append(chunk_ends, len(non_nan_mask))
        
    return list(zip(chunk_starts, chunk_ends))

def process_raw_online(raw: mne.io.Raw):
    """Process raw data using online-style preprocessing"""
    # Remove problematic channels
    raw.drop_channels(["Fp1-1"], on_missing='ignore')
    try:
        raw.rename_channels({"Fp1-0": "Fp1"})
    except Exception as e:
        print("Fp1-0 not found")
        pass
    
    # Analyze NaN chunks before preprocessing
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    total_duration = data.shape[1] / sfreq
    print(f"\nTotal signal duration before processing: {total_duration:.2f}s")
    
    # Find and crop to largest non-NaN chunk
    non_nan_chunks = find_non_nan_chunks(data)
    if non_nan_chunks:
        chunk_lengths = [(end - start) for start, end in non_nan_chunks]
        largest_chunk_idx = np.argmax(chunk_lengths)
        start_idx, end_idx = non_nan_chunks[largest_chunk_idx]
        
        print(f"Largest non-NaN chunk: {start_idx/sfreq:.2f}s to {end_idx/sfreq:.2f}s")
        raw.crop(tmin=start_idx/sfreq, tmax=(end_idx - 1)/sfreq)
    
    # Resample to target frequency
    raw.resample(DOWNSAMPLE_SFREQ)
    sfreq = raw.info["sfreq"]
    print(f'SFREQ after resampling: {sfreq}')
    
    # Apply online preprocessing
    raw = apply_online_preprocessing(raw)
    
    # Ensure we have the right channels
    raw.pick(picks=list(CHANNELS_TO_KEEP))
    
    return raw

def process_subject_data_online(subject_id):
    """Process subject data using online-style preprocessing"""
    print(f"\nProcessing subject {subject_id} with online preprocessing")
    
    # Load raw data
    protocol = ns.io.NeuroStim(
        task_file=f"{RAW_DATA_DIR}/{subject_id}/Task.json",
        xdf_file=f"{RAW_DATA_DIR}/{subject_id}/data.xdf",
        event_stream_name="Brainbuilding-Events"
    )
    raw, _ = protocol.raw_xdf(
        annotation=True,
        eeg_stream_names=["NeoRec21-1247", "actiCHamp-23090108"],
        extended_annotation=True,
        # include_vas=True,
    )
    
    # Process with online-style preprocessing
    processed_raw = process_raw_online(raw)
    
    # Save processed data
    output_path = f'{TRAINING_DATA_DIR}/{subject_id}.fif'
    processed_raw.save(output_path, overwrite=True)
    print(f"Saved online-preprocessed data to {output_path}")

def compare_preprocessing_methods(subject_id):
    """Compare online preprocessing with standard preprocessing for a single subject"""
    print(f"\nComparing preprocessing methods for subject {subject_id}")
    
    # Load standard preprocessed data
    standard_path = f'data/fif-dataset-2/{subject_id}.fif'
    if not os.path.exists(standard_path):
        print(f"Standard preprocessed data not found at {standard_path}")
        return
        
    standard_raw = mne.io.read_raw_fif(standard_path, preload=True)
    
    # Load online preprocessed data
    online_path = f'{TRAINING_DATA_DIR}/{subject_id}.fif'
    if not os.path.exists(online_path):
        print(f"Online preprocessed data not found at {online_path}")
        return
        
    online_raw = mne.io.read_raw_fif(online_path, preload=True)
    
    # Compare basic properties
    print(f"Standard preprocessing:")
    print(f"  Shape: {standard_raw.get_data().shape}")
    print(f"  Sampling rate: {standard_raw.info['sfreq']} Hz")
    print(f"  Duration: {standard_raw.times[-1]:.2f}s")
    print(f"  Channels: {standard_raw.ch_names}")
    
    print(f"Online preprocessing:")
    print(f"  Shape: {online_raw.get_data().shape}")
    print(f"  Sampling rate: {online_raw.info['sfreq']} Hz")
    print(f"  Duration: {online_raw.times[-1]:.2f}s")
    print(f"  Channels: {online_raw.ch_names}")
    
    # Compare data statistics
    standard_data = standard_raw.get_data()
    online_data = online_raw.get_data()
    
    print(f"\nData statistics comparison:")
    print(f"Standard - Mean: {np.mean(standard_data):.6f}, Std: {np.std(standard_data):.6f}")
    print(f"Online   - Mean: {np.mean(online_data):.6f}, Std: {np.std(online_data):.6f}")
    
    # If same duration and channels, compute correlation
    if (standard_data.shape == online_data.shape and 
        set(standard_raw.ch_names) == set(online_raw.ch_names)):
        
        # Reorder channels to match
        online_reordered = np.zeros_like(standard_data)
        for i, ch in enumerate(standard_raw.ch_names):
            j = online_raw.ch_names.index(ch)
            online_reordered[i] = online_data[j]
        
        # Compute correlation
        correlation = np.corrcoef(standard_data.flatten(), online_reordered.flatten())[0, 1]
        print(f"Overall correlation between methods: {correlation:.4f}")

def main():
    """Main function to run online preprocessing"""
    subject_ids = [f for f in os.listdir(RAW_DATA_DIR) if f.isdigit()]
    subject_ids = sorted(subject_ids)   
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    # Process a few subjects first for testing
    test_subjects = subject_ids  # Process first 3 subjects for testing
    
    for subject_id in test_subjects:
        process_subject_data_online(subject_id)
        compare_preprocessing_methods(subject_id)

if __name__ == "__main__":
    main()