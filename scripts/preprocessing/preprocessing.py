import warnings
import numpy as np
import mne
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
from typing import TypedDict
from config import STANDARD_EVENT_NAME_TO_ID_MAPPING

import glob
import pathlib
import os

FIF_DATA_DIR = os.getenv("FIF_DATA_DIR", 'data/fif-dataset-2')
DATASET_FNAME = os.getenv("DATASET_NAME", 'data/preprocessed/motor-imagery-2.parquet')

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
    sample_window = int(1. * sfreq)   # Window size
    sample_step = int(0.8 * sfreq)   # Step size
    
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
        X_train, y_train, sample_weights_train, event_ids_train = extract_epochs_from_raw(raw['raw_data'])
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
    
    # Create a table with metadata and arrays

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DATASET_FNAME), exist_ok=True)
    
    # # Save to parquet file
    # data = {
    #     'X': X,
    #     'y': y,
    #     'subject_ids': subject_ids,
    #     'sample_weights': sample_weights,
    #     'event_ids': event_ids
    # }
    
    # Create PyArrow table
    table = pa.Table.from_pydict({
        'y': pa.array(y),
        'subject_ids': pa.array(subject_ids),
        'sample_weights': pa.array(sample_weights),
        'event_ids': pa.array(event_ids),
        'X': pa.array(X.tolist())  # Convert numpy array to list for PyArrow
    })
    
    # Write to parquet file
    pq.write_table(table, DATASET_FNAME)

    print("Data saved successfully:")
    print(f"Shape: {X.shape}")
    print(f"Saved to: {DATASET_FNAME}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique event IDs: {len(np.unique(event_ids))}")

if __name__ == "__main__":
    main()
