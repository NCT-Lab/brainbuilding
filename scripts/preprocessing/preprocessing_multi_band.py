import warnings
import numpy as np
import mne
import pandas as pd
import h5py
import uuid
from typing import TypedDict
import sys
import glob
import pathlib
import os
import scipy.signal
from scipy.signal import butter

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.brainbuilding.config import STANDARD_EVENT_NAME_TO_ID_MAPPING


FIF_DATA_DIR = os.getenv("FIF_DATA_DIR", 'data/fif-dataset-2')
DATASET_FNAME = os.getenv("DATASET_NAME", 'data/preprocessed/motor-imagery-2.h5')

warnings.filterwarnings('ignore') 

class SubjectRaw(TypedDict):
    subject_id: int
    raw_data: mne.io.Raw

def load_training_data() -> list[SubjectRaw]:
    """Load training data from training_data directory"""
    training_files = glob.glob(f'{FIF_DATA_DIR}/*.fif')
    training_files = sorted(training_files)
    result = []
    
    for file_path in training_files:
        # Extract subject_id from filename
        subject_id = int(pathlib.Path(file_path).stem)
        # Load raw data
        raw = mne.io.read_raw_fif(file_path, preload=True)
        result.append(SubjectRaw(subject_id=subject_id, raw_data=raw))
    
    return result

def _extract_epochs_from_events(data: np.ndarray, events: np.ndarray, sfreq: float):
    """Helper method to extract epochs from events using sliding windows"""
    epochs_data = []
    labels = []
    vas = []
    event_ids = []
    
    sample_delay = int(1. * sfreq)  # Initial delay after event
    sample_window = int(3. * sfreq)   # Window size
    sample_step = int(3. * sfreq)   # Step size
    
    for ind, (ts_idx, vas_val, event_id) in enumerate(events[:-1]):  # Exclude last event
        if event_id not in [0, 1]:
            continue
        
        start_idx = ts_idx + sample_delay
        
        # search for the next event with different event_id
        next_events = [i for i in events[ind + 1:] if i[2] != event_id]
        if not next_events:
            continue
        end_idx = next_events[0][0]
        
        # Get the unique random ID for this event
        unique_event_id = np.random.randint(0, 2**64 - 1, dtype=np.uint64)
        
        chunk_start = start_idx
        chunk_end = chunk_start + sample_window
        while chunk_end <= end_idx:
            chunk = data[:, chunk_start:chunk_end]
            epochs_data.append(chunk)
            labels.append(event_id)
            vas.append(vas_val)
            event_ids.append(unique_event_id)
            chunk_start = chunk_start + sample_step
            chunk_end = chunk_start + sample_window
    
    return np.array(epochs_data), np.array(labels), np.array(vas), np.array(event_ids)

def extract_epochs_from_raw(raw: mne.io.Raw):
    """Extract epochs for Rest and Animation events using sliding windows"""
    data = raw.get_data()
    zero_event_names = []
    one_event_names = []
    
    sfreq = raw.info['sfreq']
    print(f'SFREQ: {sfreq}')

    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    events_reprocessed = []
    
    # Fixed event processing - iterate through events and look up descriptions
    for event in events:
        ts_i, _, e = event
        # Find all descriptions that map to this event ID
        descriptions = [k for k, v in event_ids.items() if v == e]
        
        # Take first description (assuming one-to-one mapping)
        desc = descriptions[0] if descriptions else ''
        
        # Extract VAS from description if present
        if "VAS=" in desc:
            vas = float(desc.split("VAS=")[1]) / 100
        else:
            vas = 1  # Default value if no VAS specified
            
        # Classify event names
        # print(desc)
        e = STANDARD_EVENT_NAME_TO_ID_MAPPING(desc)
        if e == 0:
            zero_event_names.append(desc)
        elif e == 1:
            one_event_names.append(desc)
            
        events_reprocessed.append((ts_i, vas, e))

    print(f'Zero event names: {sorted(list(set(zero_event_names)))}')
    print(f'One event names: {sorted(list(set(one_event_names)))}')
    
    epochs_data, labels, sample_weights, event_ids = _extract_epochs_from_events(data, events_reprocessed, sfreq)
    print(f'{np.unique([len(i[0]) for i in epochs_data])}')

    return epochs_data, labels, sample_weights, event_ids


def create_filtered_bands(raw):
    """Create filtered versions of each channel in different frequency bands and remove original channels"""
    # Define frequency bands
    freq_bands = {
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30)
    }
    
    new_data = []
    new_ch_names = []
    new_ch_types = []
    
    # For each frequency band
    for band_name, (low_freq, high_freq) in freq_bands.items():
        # Create a copy of the raw object for this band
        raw_band = raw.copy()
        # Apply bandpass filter using MNE
        raw_band.filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin', verbose=False)
        
        # Get the filtered data and channel names
        data = raw_band.get_data()
        for i, ch_name in enumerate(raw_band.ch_names):
            new_data.append(data[i])
            new_ch_names.append(f'{ch_name}_{band_name}')
            new_ch_types.append('eeg')
    
    # Store original channel names to drop them later
    original_channels = raw.ch_names.copy()
    
    # Add the new channels to the existing raw instance
    info = mne.create_info(new_ch_names, raw.info['sfreq'], new_ch_types)
    raw.add_channels([mne.io.RawArray(new_data, info)], force_update_info=True)
    
    # Drop the original channels
    raw.drop_channels(original_channels)
    
    return raw

def main():
    # Load processed data
    raws_train = load_training_data()
    
    # Extract epochs from all training subjects
    X = []
    y = []
    subject_ids = []
    sample_weights = []
    event_ids = []
    
    for raw in raws_train:
        # Create filtered versions of the channels
        filtered_raw = create_filtered_bands(raw['raw_data'])
        X_train, y_train, sample_weights_train, event_ids_train = extract_epochs_from_raw(filtered_raw)
        X.extend(X_train)
        y.extend(y_train)
        subject_ids.extend([raw['subject_id']] * len(y_train))
        sample_weights.extend(sample_weights_train)
        event_ids.extend(event_ids_train)
    print(f'Subject IDs: {len(subject_ids)}')
    
    X = np.array(X)
    y = np.array(y)
    subject_ids = np.array(subject_ids)
    sample_weights = np.array(sample_weights)
    event_ids = np.array(event_ids)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DATASET_FNAME), exist_ok=True)
    
    # Save to HDF5 file
    print(X.shape)
    
    with h5py.File(DATASET_FNAME, 'w') as f:
        f.create_dataset('X', data=X, compression='gzip')
        f.create_dataset('y', data=y, compression='gzip')
        f.create_dataset('subject_ids', data=subject_ids, compression='gzip')
        f.create_dataset('sample_weights', data=sample_weights, compression='gzip')
        f.create_dataset('event_ids', data=event_ids, compression='gzip')
    
    print("Data saved successfully:")
    print(f"Shape: {X.shape}")
    print(f"Saved to: {DATASET_FNAME}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique event IDs: {len(np.unique(event_ids))}")

if __name__ == "__main__":
    main()
