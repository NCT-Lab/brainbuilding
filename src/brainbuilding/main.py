import asyncio
import json
import numpy as np
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, sosfiltfilt, resample, sosfilt_zi
import multiprocessing as mp
from queue import Empty
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging
from concurrent.futures import ProcessPoolExecutor, Future
import numpy.typing as npt
from scipy import signal
# import signal
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FastICA, PCA
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
import pickle
from enum import IntEnum
import traceback
from brainbuilding.transformers import AugmentedDataset
from brainbuilding.model import CSPWithChannelSelection

DEBUG = False


CHANNELS_IN_STREAM = np.array(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz'])
REF_CHANNEL_IND = [i for i, ch in enumerate(CHANNELS_IN_STREAM) if ch == 'Fz'][0]
CHANNELS_TO_KEEP = np.array([
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Cz", "Pz"
])
CHANNELS_MASK = np.array([True if ch in CHANNELS_TO_KEEP else False for ch in CHANNELS_IN_STREAM])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)04d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class EEGConfig:
    """Configuration for EEG processing"""
    channels: List[str]
    reference_channel: str
    processing_window_size: int = 250   # Points for classification
    processing_step: int = 125          # Points between classifications
    sfreq: float = 250.0               # Sampling frequency
    
    def __post_init__(self):
        self.ref_idx = self.channels.index(self.reference_channel)
        # Channels to keep after reference subtraction (excluding reference)
        self.keep_channels = [i for i, ch in enumerate(self.channels) 
                            if ch != self.reference_channel]

# Add Event enum definition at the top level
class Event(IntEnum):
    """Event IDs from Brainbuilding-Events stream"""
    BACKGROUND = 1
    REST = 2
    PREPARE = 3
    LEFT_HAND = 4
    RIGHT_HAND = 5
    
    BLINK = 10
    TRACK_DOT = 11
    RANDOM_DOT = 12
    
    @classmethod
    def is_calibration_event(cls, event) -> bool:
        return event in [cls.BLINK, cls.TRACK_DOT, cls.RANDOM_DOT]

@dataclass
class Sample:
    """Single EEG sample with timestamp"""
    data: np.ndarray
    timestamp: float

@dataclass
class ProcessingState:
    """Current state of processing pipeline"""
    ica: Optional["EyeRemoval"] = None
    sos: Optional[npt.NDArray[np.float64]] = None
    is_training: bool = True
    ica_future: Optional[Future[Tuple["EyeRemoval", npt.NDArray[np.float64]]]] = None
    current_event: Optional[Event] = None
    collecting_calibration: bool = False
    collecting_background: bool = False
    processing_background: bool = False
    processing_calibration: bool = False
    processed_and_raw_windows: List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    background_future: Optional[Future[npt.NDArray[np.float64]]] = None
    feature_medians: Optional[npt.NDArray[np.float64]] = None

@dataclass
class WindowMetadata:
    """Metadata for window processing"""
    start_timestamp: float
    end_timestamp: float

class GracefulKiller:
    """Helper class to handle shutdown signals"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

async def tcp_sender(queue: mp.Queue, host: str, port: int):
    """Async process that handles TCP connection and sending results"""
    writer = None
    retries = 0
    max_retries = 500
    
    while retries < max_retries:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            logger.info("TCP connection established")
            break
        except ConnectionRefusedError:
            retries += 1
            logger.warning(f"Connection attempt {retries} failed, retrying in 1 second...")
            await asyncio.sleep(1)
    
    if writer is None:
        logger.error("Failed to establish TCP connection")
        return
        
    try:
        while True:
            try:
                result= queue.get_nowait()
                message = result
                data = json.dumps(message).encode() + b"\n"
                writer.write(data)
                await writer.drain()
                logger.info(f"Sent classification result: {result}")
            except Empty:
                await asyncio.sleep(0.0001)
            except Exception as e:
                logger.error(f"Error sending result: {e}")
                await asyncio.sleep(0.1)
                
    except asyncio.CancelledError:
        logger.info("TCP sender received shutdown signal")
    except Exception as e:
        logger.error(f"TCP sender error: {e}")
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        logger.info("TCP sender shutting down")

def tcp_sender_process(queue: mp.Queue, host: str, port: int):
    """Process that runs the TCP sender"""
    asyncio.run(tcp_sender(queue, host, port))

def load_pipeline() -> Tuple[Pipeline, CSPWithChannelSelection]:
    """Load trained pipeline and CSP from pickle file"""
    try:
        with open('trained_pipeline.pickle', 'rb') as f:
            data = pickle.load(f)
            pipeline = data['pipeline']
            csp = data['csp']
        logger.info("Classification pipeline and CSP loaded successfully")
        return pipeline, csp
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise

def fit_ica(raw_data: List, timestamps: List) -> Tuple['EyeRemoval', np.ndarray]:
    """Fit ICA and create filter (runs in ProcessPoolExecutor)"""
    # Convert to numpy arrays
    filtered_data = np.array(raw_data).T
    timestamps = np.array(timestamps)
    
    # Create filter
    veog, heog = create_standard_eog_channels(filtered_data)
    
    # Create and fit ICA
    ica = EyeRemoval(n_components=len(CHANNELS_TO_KEEP), veog_data=veog, heog_data=heog)
    ica.fit(filtered_data)
    
    return ica

class EyeRemoval(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, veog_data, heog_data, random_state=42):
        self.ica = FastICA(n_components=n_components, random_state=random_state)
        self.veog_data = veog_data
        self.heog_data = heog_data
        
    def fit(self, X, y=None):
        X_transformed = self.ica.fit_transform(X.T).T
        veog_correlations = np.array([stats.pearsonr(source, self.veog_data)[0] for source in X_transformed])
        heog_correlations = np.array([stats.pearsonr(source, self.heog_data)[0] for source in X_transformed])
        self.veog_idx = np.argmax(np.abs(veog_correlations))
        self.heog_idx = np.argmax(np.abs(heog_correlations))
        return self
    
    def transform(self, X):
        X_transformed = self.ica.transform(X.T)
        X_transformed[:, self.veog_idx] = 0
        X_transformed[:, self.heog_idx] = 0
        X_transformed = self.ica.inverse_transform(X_transformed).T
        return X_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_sources(self, X):
        """Get ICA sources for benchmarking"""
        return self.ica.transform(X.T).T
    
    def score_sources(self, X, target, start=None, stop=None):
        """Score ICA components against a target channel"""

        sources = self.get_sources(X)
        if isinstance(target, (int, np.integer)):
            target = X[target, start:stop]
        else:
            target = target[start:stop]
            
        correlations = np.array([stats.pearsonr(source, target)[0] for source in sources])
        return correlations

def create_bandpass_filter(sfreq: float):
    """Create bandpass filter coefficients"""
    nyq = sfreq / 2
    low = 4.0 / nyq
    high = 40.0 / nyq
    return butter(4, [low, high], btype='band', output='sos')

# def process_point(data: np.ndarray, _config: EEGConfig) -> np.ndarray:
#     """Process a single data point"""
#     # Subtract reference channel
#     referenced = data - data[REF_CHANNEL_IND]
#     # Convert from µV to V
#     referenced = referenced / 1000000
#     # Keep only needed channels
#     return referenced[CHANNELS_MASK]

def filter_window(data: np.ndarray, sos) -> np.ndarray:
    """Apply bandpass filter to window"""
    return sosfiltfilt(sos, data, axis=-1)

def create_standard_eog_channels(
    data: npt.NDArray[np.float64], 
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create vEOG and hEOG channels from EEG data"""
    # Get indices for EOG computation
    fp1 = data[CHANNELS_TO_KEEP == 'Fp1'][0]
    fp2 = data[CHANNELS_TO_KEEP == 'Fp2'][0]
    f7 = data[CHANNELS_TO_KEEP == 'F7'][0]
    f8 = data[CHANNELS_TO_KEEP == 'F8'][0]
    
    # Compute EOG channels
    veog = (fp2 + fp1) / 2
    heog = f8 - f7
    
    return veog, heog

def process_background_data(
    raw_data: List[List[float]], 
    timestamps: List[float],
    ica: EyeRemoval,
    window_size: int,
    csp: CSPWithChannelSelection,
) -> npt.NDArray[np.float64]:
    """Process background data to compute covariance bounds and CSP feature medians"""
    logger.info(f"Processing background data with {len(raw_data)} samples")
    
    try:
        # Convert to numpy arrays
        processed_data = np.array(raw_data).T
        
        # Apply ICA
        cleaned = ica.transform(processed_data)
        
        # Split into windows
        n_windows = (cleaned.shape[1] - window_size) // window_size + 1
        windows = np.array([
            cleaned[:, i*window_size:(i+1)*window_size] 
            for i in range(n_windows)
        ])
        
        # Compute covariances
        # Compute CSP features for each window
        augmentation = AugmentedDataset(4, 8)
    
        X_aug = augmentation.transform(windows)
        X_cov = Covariances(estimator='oas').transform(X_aug)
        csp_features = csp.transform(X_cov)
        
        # Compute median features
        feature_medians = np.median(csp_features, axis=0, keepdims=True)
        
        logger.info(f"Background processing complete. Feature medians: {feature_medians}")
        return feature_medians
        
    except Exception as e:
        logger.error(f"Error processing background data: {e}", exc_info=True)
        raise

class SignalFilter:
    """Stateful bandpass filter for EEG data"""
    def __init__(self, sfreq: float, n_channels: int):
        self.sos = create_bandpass_filter(sfreq)
        # Initialize filter state for each channel
        self.zi = np.array([sosfilt_zi(self.sos) for _ in range(n_channels)])
        self.zi = np.rollaxis(self.zi, 0, 3)
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Process single sample or batch of samples"""
        filtered, self.zi = signal.sosfilt(self.sos, data, axis=0, zi=self.zi)
        return filtered

class EEGProcessor:
    def __init__(self, config: EEGConfig, tcp_host: str, tcp_port: int):
        self.config = config
        
        # Load classification pipeline and CSP
        self.pipeline, self.csp = load_pipeline()
        
        # Raw data storage
        self.raw_samples: List[List[float]] = []
        self.timestamps: List[float] = []
        self.points_since_last_process: int = 0

        self.queue = mp.Queue()
        
        # Processing state
        self.state: ProcessingState = ProcessingState()
        self.state.processed_and_raw_windows = []
        
        # Start TCP sender process
        self.tcp_sender = mp.Process(
            target=tcp_sender_process,
            args=(self.queue, tcp_host, tcp_port)  # Use global queue
        )
        self.tcp_sender.start()
        
        # Create process pool for all processing
        self.executor = ProcessPoolExecutor()

        # Initialize calibration data storage
        self.calibration_data: List[List[float]] = []
        self.calibration_timestamps: List[float] = []

        # Add background data storage
        self.background_data: List[List[float]] = []
        self.background_timestamps: List[float] = []

        # Add signal filter
        self.signal_filter = SignalFilter(
            sfreq=config.sfreq,
            n_channels=len(CHANNELS_TO_KEEP)
        )
        
        # Add processed data storage
        self.processed_samples: List[np.ndarray] = []

    def update_event_state(self, events: List[List[float]]) -> None:
        """Update processing state based on received events"""
        if not events:  # Empty list means previous event continues
            return
            
        # Get the latest event
        event_id = int(events[-1][1])
        try:
            new_event = Event(event_id)
        except ValueError:
            logger.warning(f"Received unknown event ID: {event_id}")
            return
            
        # Update state
        if self.state.current_event != new_event:
            logger.info(f"Event changed: {new_event.name}")
            self.state.current_event = new_event
            
            if new_event == Event.BACKGROUND:
                if not self.state.collecting_background:
                    logger.info("Starting background data collection")
                    self.state.collecting_background = True
                    self.state.cov_bounds = None
                    self.state.processing_background = False
                    self.background_data.clear()
                    self.background_timestamps.clear()
            else:
                # If we were collecting background, process it
                if self.state.collecting_background:
                    logger.info("Background collection complete, starting processing")
                    self.state.collecting_background = False
                    self.state.processing_background = True

            
            # Handle calibration events
            if Event.is_calibration_event(new_event):
                if not self.state.collecting_calibration:
                    logger.info("Starting calibration data collection")
                    self.state.collecting_calibration = True
                    self.state.ica = None
                    self.state.processing_calibration = False
                    self.calibration_data.clear()
                    self.calibration_timestamps.clear()
            else:
                if self.state.collecting_calibration:
                    logger.info("Calibration collection complete, starting ICA fitting")
                    self.state.collecting_calibration = False
                    self.state.processing_calibration = True

    def _start_background_processing(self) -> None:
        """Start background data processing"""
        if self.state.ica is None:
            logger.warning("Cannot process background data: ICA not fitted yet")
            return
            
        logger.info(f"Starting background processing with {len(self.background_data)} samples")
        
        self.state.background_future = self.executor.submit(
            process_background_data,
            self.background_data,
            self.background_timestamps,
            self.state.ica,
            self.config.processing_window_size,
            self.csp
        )

    def _start_ica_fitting(self) -> None:
        """Start ICA fitting using collected calibration data"""
        logger.info(f"Starting ICA fitting with {len(self.calibration_data)} samples")
        
        self.state.ica_future = self.executor.submit(
            fit_ica, 
            self.calibration_data,
            self.calibration_timestamps
        )
        logger.info("ICA fitting started with calibration data")

    def process_raw_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process a single raw sample"""
        # Subtract reference channel
        referenced = sample - sample[:, REF_CHANNEL_IND:REF_CHANNEL_IND+1]
        
        # Convert from µV to V
        scaled = referenced / 1_000_000
        
        # Keep only needed channels
        selected = scaled[:, CHANNELS_MASK]
        
        # Apply bandpass filter
        filtered = self.signal_filter.process(selected)
        
        return filtered

    async def process_sample(self, samples, timestamps: float) -> None:
        """Process and store samples"""
        # Process each sample
        processed = []
        processed_samples = self.process_raw_sample(np.array(samples))
        processed.extend(processed_samples.tolist())
            
        # Store raw and processed samples
        self.raw_samples += samples
        self.processed_samples += processed
        self.timestamps += timestamps
        
        # Store in appropriate buffer based on current state
        if self.state.collecting_calibration:
            self.calibration_data += processed
            self.calibration_timestamps += timestamps
        elif self.state.collecting_background:
            self.background_data += processed
            self.background_timestamps += timestamps
        
        # Check futures
        if self.state.processing_calibration:
            if self.state.ica_future is None:
                logger.info(f'Sample len: {len(self.calibration_data[0])}')
                self._start_ica_fitting()
            else:
                if self.state.ica_future.done():
                    try:
                        self.state.ica = self.state.ica_future.result(timeout=0)
                        self.state.processing_calibration = False
                        self.state.ica_future = None
                        logger.info("ICA fitting completed successfully")
                    except Exception as e:
                        logger.error(f"ICA fitting failed: {e}\n{traceback.format_exc()}")
                        self.state.ica_future = None
        
        if self.state.processing_background:
            if self.state.background_future is None:
                self._start_background_processing()
            else:
                if self.state.background_future.done():
                    try:
                        feature_medians = self.state.background_future.result(timeout=0)
                        self.state.feature_medians = feature_medians
                        self.state.background_future = None
                        self.state.processing_background = False
                        logger.info(f"Background processing complete. Feature medians: {feature_medians}")
                    except Exception as e:
                        logger.error(f"Background processing failed: {e}")
                        self.state.background_future = None
                    
        # Process window if ready
        if self.state.ica is not None and self.state.feature_medians is not None:
            self.points_since_last_process += len(samples)
            
            if self.points_since_last_process >= self.config.processing_step:
                self.points_since_last_process = 0
                if len(self.timestamps) >= self.config.processing_window_size:
                    self.process_window()

    def process_window(self) -> None:
        """Submit window for processing if ICA is ready"""
        window_data = self.processed_samples[-self.config.processing_window_size:]
        window_timestamps = self.timestamps[-self.config.processing_window_size:]
        
        metadata = WindowMetadata(
            start_timestamp=window_timestamps[0],
            end_timestamp=window_timestamps[-1]
        )
        
        future = self.executor.submit(
            process_single_window,
            window_data,
            window_timestamps,
            metadata,
            self.state.ica,
            self.pipeline,
            self.csp,
            self.state.feature_medians
        )
        future.add_done_callback(self.process_window_callback)

    def process_window_callback(self, future: Future) -> None:
        """Callback for window processing"""
        result, processed_data_copy, cleaned_copy = future.result()
        self.queue.put(result)
        if DEBUG:
            self.state.processed_and_raw_windows.append((processed_data_copy, cleaned_copy))

    def save_history(self, filename: str) -> None:
        """Save complete sample history to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'raw_samples': self.raw_samples,
                'timestamps': self.timestamps,
                'processed_and_raw_windows': self.state.processed_and_raw_windows
            }, f)

    async def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Starting cleanup...")
        
        # Shutdown process pool first
        logger.info("Shutting down process pool...")
        self.executor.shutdown(wait=True, cancel_futures=True)
        
        # Terminate TCP sender
        logger.info("Terminating TCP sender...")
        if self.tcp_sender.is_alive():
            self.tcp_sender.terminate()
            self.tcp_sender.join(timeout=1.0)
        
        # Save history
        logger.info("Saving history...")
        self.save_history('eeg_history.pickle')
        
        logger.info("Cleanup completed")

def process_single_window(
    processed_data: List[List[float]], 
    timestamps: List[float],
    metadata: WindowMetadata, 
    ica: EyeRemoval, 
    pipeline: Pipeline,
    csp: CSPWithChannelSelection,
    feature_medians: npt.NDArray[np.float64]
) -> Optional[Dict[str, float]]:
    """Process a single window of preprocessed EEG data"""
    logger.info("Starting window inner processing...")
    try:
        start_time = time.time()
        
        # Convert lists to numpy arrays
        data = np.array(processed_data)
        data = data.T  # Convert to (channels, samples) format
        
        # Apply ICA
        cleaned = ica.transform(data)
        cleaned_copy = cleaned.copy()
        
        # Reshape for classification
        X = cleaned.reshape(1, *cleaned.shape)
        
        # Compute covariance
        augmentation = AugmentedDataset(order=4, lag=8)
        X_aug = augmentation.transform(X)
        X_cov = Covariances(estimator='oas').transform(X_aug)
        
        # Apply CSP and normalize features
        csp_features = csp.transform(X_cov)
        # normalized_features = (csp_features - feature_medians) / feature_medians
        normalized_features = csp_features
        
        # Classify using normalized features
        prediction = pipeline.predict(normalized_features)[0]
        probabilities = pipeline.predict_proba(normalized_features)[0]
        
        classification_time = time.time() - start_time
        
        result = {
            "class": int(prediction),
            "probability": float(probabilities[1]),
            "window_start_lsl_timestamp": float(metadata.start_timestamp),
            "window_end_lsl_timestamp": float(metadata.end_timestamp),
            "classification_time_in_sec": float(classification_time)
        }
        
        logger.info(f"Window processing completed in {classification_time*1000:.2f}ms")
        logger.info(f"Classification result: class={result['class']}, "
                   f"probability={result['probability']:.3f}")
        
        return result, data, cleaned_copy
        
    except Exception as e:
        logger.error(f"Error processing window: {e}", exc_info=True)
        return None

async def process_eeg_stream(config: EEGConfig, tcp_host: str, tcp_port: int):
    """Main processing function"""
    processor = EEGProcessor(config, tcp_host, tcp_port)
    
    try:
        # Find both streams
        eeg_streams = resolve_stream('name', 'NeoRec21-1247')
        event_streams = resolve_stream('name', 'Brainbuilding-Events')
        
        if not eeg_streams or not event_streams:
            raise RuntimeError("Could not find required streams")
            
        eeg_inlet = StreamInlet(eeg_streams[0])
        event_inlet = StreamInlet(event_streams[0])
        
        while True:
            try:
                # Get new events (timeout=0 for non-blocking)
                events, event_timestamps = event_inlet.pull_chunk(timeout=0)
                processor.update_event_state(events)
                
                # Get new EEG data
                samples, timestamps = eeg_inlet.pull_chunk()
                if len(samples) == 0:
                    continue
                
                await processor.process_sample(samples, timestamps)
                await asyncio.sleep(0)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
                    
    except Exception as e:
        logger.error(f"Error in main processing loop: {e}")
    finally:
        logger.info("Starting shutdown sequence...")
        await processor.cleanup()
        logger.info("Shutdown complete")

def main():
    # Configuration
    config = EEGConfig(
        channels=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                 "F7", "F8", "T3", "T4", "T5", "T6", "Cz", "Fz"],
        reference_channel="Fz"
    )
    
    try:
        # Run processing
        asyncio.run(process_eeg_stream(config, "192.168.1.97", 8080))
    except KeyboardInterrupt:
        logger.info("Main process received keyboard interrupt")

if __name__ == "__main__" and mp.current_process().name == 'MainProcess':
    main() 