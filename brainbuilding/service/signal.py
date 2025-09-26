import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


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
        return butter(4, [low, high], btype="band", output="sos")

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process single sample maintaining filter state"""
        sample = sample.reshape(1, -1)  # Ensure correct shape
        filtered, self.zi = sosfilt(self.sos, sample, axis=0, zi=self.zi)
        return filtered.flatten()

    def process_samples_batch(self, samples: np.ndarray) -> np.ndarray:
        filtered, self.zi = sosfilt(self.sos, samples, axis=0, zi=self.zi)
        return filtered
