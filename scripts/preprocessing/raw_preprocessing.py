import warnings
import pyneurostim as ns
import numpy as np
import mne
from scipy.signal import butter
import scipy.signal
import sys
import os
from mne.preprocessing import ICA
# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from brainbuilding.core.transformers import EyeRemoval
from src.brainbuilding.eye_removal import create_standard_eog_channels
from brainbuilding.core.config import PICK_CHANNELS, ND_CHANNELS_MASK, ORDER, REMOVE_HEOG, REMOVE_VEOG, LOW_FREQ, HIGH_FREQ
import os

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", 'data/raw-dataset-2/')
TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", 'data/fif-dataset-2')

warnings.filterwarnings('ignore')
DOWNSAMPLE_SFREQ = 250 

def filtering_method(signal, sfreq=500.0):
    # Apply filter
    nyq = sfreq / 2
    low = LOW_FREQ / nyq
    high = HIGH_FREQ / nyq
    sos = butter(4, [low, high], btype='band', output='sos')
    filtered = scipy.signal.sosfilt(sos, signal, axis=-1)
    return filtered

def apply_bandpass_filter(raw, sfreq=500.0):
    # Find first and last non-nan indices
    data_to_process, start_idx, end_idx = get_notna_data(raw.get_data().copy())
    filtered = filtering_method(data_to_process, sfreq)
    
    # Insert filtered signal back
    raw._data[:, start_idx:end_idx] = filtered
    return raw

def get_notna_data(data):
    non_nan_indices = np.where(~np.isnan(data))[1]
    if len(non_nan_indices) == 0:
        return data, 0, data.shape[-1]
        
    start_idx = non_nan_indices[0]
    end_idx = non_nan_indices[-1]
    
    # Get data segment for processing
    return data[:, start_idx:end_idx], start_idx, end_idx

def apply_ica(raw):
    raw_data = raw.get_data()
    print(f"{np.all(raw_data == raw.get_data())=}")
    ica = ICA(n_components=18, max_iter='auto')
    ica.fit(raw, verbose=True)
    # ica.exclude = [0,1]
    eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0, ch_name=['Fp1', 'Fp2'], verbose=False)
    print(f"EOG indices: {eog_indices}")
    print(f"EOG scores: {eog_scores}")
    ica.exclude = eog_indices
    ica.apply(raw)
    new_raw_data = raw.get_data()
    print(f"{np.all(raw_data == new_raw_data)=}")
    # data = raw.get_data()
    # result = data.copy()
    # data_to_process, start_idx, end_idx = get_notna_data(data)
    
    # # Create EOG channels
    # veog_data, heog_data = create_standard_eog_channels(data_to_process)
    # # Apply EyeRemoval ICA
    # eye_removal = EyeRemoval(n_components=18, 
    #                         veog_data=veog_data,
    #                         heog_data=heog_data,
    #                         remove_veog=REMOVE_VEOG,
    #                         remove_heog=REMOVE_HEOG)
    
    # eye_removal.fit(data_to_process)
    # # Clean the data
    # data_cleaned = eye_removal.transform(data_to_process)
    
    # # Insert cleaned data back into original array
    # result[:, start_idx:end_idx] = data_cleaned
    # raw._data = result
    
    return raw

def extract_epochs(raw_data, events):
    """Extract epochs for Rest and Animation events using sliding windows"""
    epochs_data = {'Rest': [], 'Animation': []}
    
    sfreq = raw_data.info["sfreq"]
    sample_250ms = int(1. * sfreq)  # Initial delay after event
    sample_500ms = int(1. * sfreq)   # Window size
    sample_100ms = int(0.8 * sfreq)   # Step size
    data = raw_data.get_data()

    for i, event in enumerate(events):  # Exclude last event to safely check next event
        event_time = event["index"]
        event_type = event["sample_type"]
        event_trial = event['trial_type']
        event_block = event['block_type']
        
        if event_type in epochs_data and event_block == 'imagin/move/anim' and event_trial in ['left/hand', 'right/hand']:  # Only process Rest and Animation events
            start_idx = event_time + sample_250ms
            end_idx = events[i + 1]["index"]  # Next event time
            
            # Calculate number of possible windows
            n_windows = (end_idx - start_idx - sample_500ms) // sample_100ms + 1
            
            chunks = []
            for j in range(n_windows):
                chunk_start = start_idx + (j * sample_100ms)
                chunk_end = chunk_start + sample_500ms
                
                # Only include complete windows
                if chunk_end <= end_idx:
                    chunk = data[:, chunk_start:chunk_end]
                    chunks.append(chunk)

            if chunks:
                epochs_data[event_type].append(np.array(chunks))

    return epochs_data

def find_nan_chunks(data):
    """Find consecutive chunks of NaN values in the signal."""
    # Get boolean mask of NaN values (True where NaN)
    nan_mask = np.any(np.isnan(data), axis=0)  # Using first channel as reference
    
    # Find where NaN status changes
    change_points = np.diff(nan_mask.astype(int))
    # Get indices where chunks start (1) and end (-1)
    chunk_starts = np.where(change_points == 1)[0] + 1
    chunk_ends = np.where(change_points == -1)[0] + 1
    
    # Handle edge cases
    if nan_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if nan_mask[-1]:
        chunk_ends = np.append(chunk_ends, len(nan_mask))
        
    return list(zip(chunk_starts, chunk_ends))

def find_non_nan_chunks(data):
    """Find consecutive chunks of non-NaN values in the signal."""
    # Get boolean mask of non-NaN values (True where not NaN)
    non_nan_mask = np.all(~np.isnan(data), axis=0)  # Using first channel as reference
    
    # Find where non-NaN status changes
    change_points = np.diff(non_nan_mask.astype(int))
    # Get indices where chunks start (1) and end (-1)
    chunk_starts = np.where(change_points == 1)[0] + 1
    chunk_ends = np.where(change_points == -1)[0] + 1
    
    # Handle edge cases
    if non_nan_mask[0]:
        chunk_starts = np.insert(chunk_starts, 0, 0)
    if non_nan_mask[-1]:
        chunk_ends = np.append(chunk_ends, len(non_nan_mask))
        
    return list(zip(chunk_starts, chunk_ends))

def process_raw(raw: mne.io.Raw):
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
    
    nan_chunks = find_nan_chunks(data)
    print("\nNaN chunks analysis before preprocessing:")
    print(f"Number of NaN chunks: {len(nan_chunks)}")
    print("\nChunk details:")
    for i, (start, end) in enumerate(nan_chunks, 1):
        duration = (end - start) / sfreq
        start_time = start / sfreq
        end_time = end / sfreq
        print(f"Chunk {i}:")
        print(f"  Start time: {start_time:.2f}s")
        print(f"  End time: {end_time:.2f}s")
        print(f"  Duration: {duration:.2f}s")
    
    # Find non-NaN chunks and get the largest one
    non_nan_chunks = find_non_nan_chunks(data)
    if non_nan_chunks:
        chunk_lengths = [(end - start) for start, end in non_nan_chunks]
        largest_chunk_idx = np.argmax(chunk_lengths)
        start_idx, end_idx = non_nan_chunks[largest_chunk_idx]
        
        print("\nLargest non-NaN chunk:")
        print(f"Start time: {start_idx/sfreq:.2f}s")
        print(f"End time: {end_idx/sfreq:.2f}s")
        print(f"Duration: {(end_idx-start_idx)/sfreq:.2f}s")
        
        # Crop to largest non-NaN chunk
        raw.crop(tmin=start_idx/sfreq, tmax=(end_idx - 1)/sfreq)
        print(f"Signal duration after cropping: {(end_idx-start_idx)/sfreq:.2f}s")
    print("")
    
    print("####")
    print('BEFORE RESAMPLING')
    _, start_idx, end_idx = get_notna_data(raw.get_data())
    print(f'{start_idx/sfreq=}')
    print(f'{end_idx/sfreq=}')
    print(f'{np.mean(np.isnan(raw.get_data()))=}')
    print("####")
    print("")
    
    raw.set_eeg_reference(ref_channels=['Fz'], verbose='error')
    raw.pick(picks=PICK_CHANNELS)
    assert raw.get_data().shape[0] == len(PICK_CHANNELS)
    
    raw.resample(DOWNSAMPLE_SFREQ)
    _, start_idx, end_idx = get_notna_data(raw.get_data())
    sfreq = raw.info["sfreq"]
    raw.crop(tmin=(start_idx + 1) / sfreq, tmax=(end_idx - 1) / sfreq)
    print("")
    print("####")
    print(f'{np.mean(np.isnan(raw.get_data()))=}')
    print("####")
    print("")
    
    sfreq = raw.info["sfreq"]
    print(f'SFREQ: {sfreq}')
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design='firwin')
    # raw = apply_bandpass_filter(raw, sfreq)
    raw = apply_ica(raw)
    raw.pick(picks=PICK_CHANNELS)
    # print(mne.events_from_annotations(raw)[1])

def process_subject_data(subject_id):
    print("")
    print(f"Processing subject {subject_id}")
    # Redirect stdout temporarily
    protocol = ns.io.NeuroStim(
        f"{RAW_DATA_DIR}/{subject_id}/Task.json",
        f"{RAW_DATA_DIR}/{subject_id}/data.xdf",
        "Brainbuilding-Events"
    )
    raw, _ = protocol.raw_xdf(
        annotation=True,
        eeg_stream_names=["NeoRec21-1247", "actiCHamp-23090108"],
        extended_annotation=True,
        include_vas=True,
    )
    preprocess_subject_events(raw)
    process_raw(raw)
    raw.save(f'{TRAINING_DATA_DIR}/{subject_id}.fif', overwrite=True)

def preprocess_subject_events(raw: mne.io.Raw):
    pass
    # print("Events:")
    # print([i for i in raw.annotations])
    # raw.annotations.rename({
    #     "Point@@left/hand": "Animation@imagin/move/anim@left/hand",
    #     "Point@@right/hand": "Animation@imagin/move/anim@right/hand",
    #     "Image@@left/hand": "Animation@imagin/move/anim@left/hand",
    #     "Image@@right/hand": "Animation@imagin/move/anim@right/hand",
    #     "Rest@@left/hand": "Rest@imagin/move/anim@left/hand",
    #     "Rest@@right/hand": "Rest@imagin/move/anim@right/hand",
    # })

def main():
    subject_ids = [f for f in os.listdir(RAW_DATA_DIR) if f.isdigit()]
    subject_ids = sorted(subject_ids)   
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    for subject_id in subject_ids:
        process_subject_data(subject_id)


if __name__ == "__main__":
    main()
