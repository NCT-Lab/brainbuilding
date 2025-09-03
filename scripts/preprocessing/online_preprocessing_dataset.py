"""
Generate dataset using online preprocessing and compare evaluation results
with the standard preprocessing method.
"""
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

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from brainbuilding.core.config import STANDARD_EVENT_NAME_TO_ID_MAPPING, BACKGROUND_CLASS_INDEX, IMAGERY_CLASS_INDEX, RIGHT_HAND_CLASS_INDEX, POINT_CLASS_INDEX

# Use online-preprocessed data
FIF_DATA_DIR = os.getenv("FIF_DATA_DIR", 'data/fif-dataset-online-preprocessed')
DATASET_FNAME = os.getenv("DATASET_NAME", 'data/preprocessed/motor-imagery-online.npy')

warnings.filterwarnings('ignore') 

class SubjectRaw(TypedDict):
    subject_id: int
    raw_data: mne.io.Raw

def load_training_data() -> list[SubjectRaw]:
    """Load training data from online-preprocessed directory"""
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
    
    # Track if we've seen a background event
    seen_background = False
    
    for ind, (ts_idx, vas_val, event_id) in enumerate(events[:-1]):  # Exclude last event
        if event_id == []:
            continue
            
        # Skip background events if we've already seen one
        if event_id[BACKGROUND_CLASS_INDEX] == 1 and seen_background:
            continue
            
        start_idx = ts_idx + sample_delay
        
        # search for the next event with different event_id
        next_events = [i for i in events[ind + 1:] if tuple(i[2]) != tuple(event_id)]
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
            
        # Mark that we've seen a background event if this was one
        if event_id[BACKGROUND_CLASS_INDEX] == 1:
            seen_background = True
    
    return np.array(epochs_data), np.array(labels), np.array(vas), np.array(event_ids)

def extract_epochs_from_raw(raw: mne.io.Raw):
    """Extract epochs for Rest and Animation events using sliding windows"""
    data = raw.get_data()
    
    sfreq = raw.info['sfreq']
    print(f'SFREQ: {sfreq}')

    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    events_reprocessed = []
    
    descs = []
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
        e = STANDARD_EVENT_NAME_TO_ID_MAPPING(desc.split("#")[0])
        descs.append(desc.split("#")[0])
        
        events_reprocessed.append((ts_i, vas, e))
    
    print(np.unique(descs))
    
    epochs_data, labels, sample_weights, event_ids = _extract_epochs_from_events(data, events_reprocessed, sfreq)
    print(f'{np.unique([len(i[0]) for i in epochs_data])}')

    return epochs_data, labels, sample_weights, event_ids

def main():
    """Generate online-preprocessed dataset"""
    print("Generating dataset from online-preprocessed data...")
    
    # Load processed data
    raws_train = load_training_data()
    
    if not raws_train:
        print(f"No data found in {FIF_DATA_DIR}. Please run online_preprocessing.py first.")
        return
    
    # Extract epochs from all training subjects
    X = []
    y = []
    is_background = []
    subject_ids = []
    sample_weights = []
    event_ids = []
    is_right_hand = []
    is_point = []
    
    for raw in raws_train:
        X_train, labels, sample_weights_train, event_ids_train = extract_epochs_from_raw(raw['raw_data'])
        y_train = labels[:, IMAGERY_CLASS_INDEX]
        X.extend(X_train)
        y.extend(y_train)
        is_background.extend(labels[:, BACKGROUND_CLASS_INDEX].astype(np.bool))
        is_right_hand.extend(labels[:, RIGHT_HAND_CLASS_INDEX].astype(np.bool))
        is_point.extend(labels[:, POINT_CLASS_INDEX].astype(np.bool))
        subject_ids.extend([raw['subject_id']] * len(y_train))
        sample_weights.extend(sample_weights_train)
        event_ids.extend(event_ids_train)
    print(f'Subject IDs: {len(subject_ids)}')
    
    X = np.array(X)
    y = np.array(y)
    print(np.unique(y))
    subject_ids = np.array(subject_ids)
    sample_weights = np.array(sample_weights)
    event_ids = np.array(event_ids)
    is_background = np.array(is_background)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DATASET_FNAME), exist_ok=True)
    
    # Create structured array
    dtype = [
        ('sample', np.float64, X.shape[1:]),
        ('label', np.int64),
        ('subject_id', np.int64),
        ('sample_weight', np.float64),
        ('event_id', np.uint64),
        ('is_background', np.bool_),
        ('is_right_hand', np.bool_),
        ('is_point', np.bool_),
    ]
    
    data = np.zeros(len(X), dtype=dtype)
    data['sample'] = X
    data['label'] = y
    data['subject_id'] = subject_ids
    data['sample_weight'] = sample_weights
    data['event_id'] = event_ids
    data['is_background'] = is_background
    data['is_right_hand'] = is_right_hand
    data['is_point'] = is_point
    # Save to numpy file
    np.save(DATASET_FNAME, data)
    
    print("Online-preprocessed data saved successfully:")
    print(f"Shape: {X.shape}")
    print(f"Saved to: {DATASET_FNAME}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique event IDs: {len(np.unique(event_ids))}")
    print(f"Number of background samples: {np.sum(is_background)}")

if __name__ == "__main__":
    main()