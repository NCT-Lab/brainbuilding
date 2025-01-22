import warnings
import numpy as np
import mne
from typing import TypedDict
from config import STANDARD_EVENT_NAME_TO_ID_MAPPING

import glob
import pathlib

warnings.filterwarnings('ignore') 

class SubjectRaw(TypedDict):
    subject_id: int
    raw_data: mne.io.Raw

def load_training_data() -> list[SubjectRaw]:
    """Load training data from training_data directory"""
    training_files = glob.glob('training_data/*.fif')
    training_files = sorted(training_files)
    result = []
    
    for file_path in training_files:
        # Extract subject_id from filename
        subject_id = int(pathlib.Path(file_path).stem)
        # Load raw data
        raw = mne.io.read_raw_fif(file_path, preload=True)
        result.append(SubjectRaw(subject_id=subject_id, raw_data=raw))
    
    return result

def load_validation_data() -> list[SubjectRaw]:
    """Load validation data from validation_data directory"""
    validation_files = glob.glob('validation_data/*.fif')
    result = []
    
    for file_path in validation_files:
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
    
    sample_delay = int(1. * sfreq)  # Initial delay after event
    sample_window = int(1. * sfreq)   # Window size
    sample_step = int(0.8 * sfreq)   # Step size
    
    for ind, (ts_idx, _, event_id) in enumerate(events[:-1]):  # Exclude last event
        if event_id not in [0, 1]:
            continue
        
        start_idx = ts_idx + sample_delay
        
        # search for the next event with different event_id
        end_idx = [i for i in events[ind + 1:] if i[2] != event_id][0][0]
        
        chunk_start = start_idx
        chunk_end = chunk_start + sample_window
        while chunk_end <= end_idx:
            chunk = data[:, chunk_start:chunk_end]
            epochs_data.append(chunk)
            labels.append(event_id)
            chunk_start = chunk_start + sample_step
            chunk_end = chunk_start + sample_window
    
    return np.array(epochs_data), np.array(labels)

def extract_epochs_from_raw(raw: mne.io.Raw):
    """Extract epochs for Rest and Animation events using sliding windows"""
    data = raw.get_data()
    zero_event_names = []
    one_event_names = []
    
    sfreq = raw.info['sfreq']
    print(f'SFREQ: {sfreq}')

    events, event_ids = mne.events_from_annotations(raw, event_id=STANDARD_EVENT_NAME_TO_ID_MAPPING, verbose=False)
    for k, v in event_ids.items():
        if v == 0:
            zero_event_names.append(k)
        elif v == 1:
            one_event_names.append(k)
    print(f'Zero event names: {sorted(list(set(zero_event_names)))}')
    print(f'One event names: {sorted(list(set(one_event_names)))}')
    
    epochs_data, labels = _extract_epochs_from_events(data, events, sfreq)
    print(f'{np.unique([len(i[0]) for i in epochs_data])}')

    return epochs_data, labels

def main():
    # Load processed data
    raws_train = load_training_data()
    
    # Extract epochs from all training subjects
    X = []
    y = []
    subject_ids = []
    for raw in raws_train:
        X_train, y_train = extract_epochs_from_raw(raw['raw_data'])
        X.extend(X_train)
        y.extend(y_train)
        subject_ids.extend([raw['subject_id']] * len(y_train))
    print(f'Subject IDs: {len(subject_ids)}')
    
    X = np.array(X)
    y = np.array(y)
    subject_ids = np.array(subject_ids)
    
    # Save the arrays
    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('subject_ids.npy', subject_ids)
    
    print("Data saved successfully:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

if __name__ == "__main__":
    main()
