import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import time
from typing import List

class StatefulSOSFilter:
    def __init__(self, num_channels: int, sfreq: float = 500.0):
        # Design filter
        nyq = sfreq / 2
        low = 4.0 / nyq
        high = 40.0 / nyq
        self.sos = butter(4, [low, high], btype='band', output='sos')
        
        # Initialize state for each channel
        self.zi = np.array([sosfilt_zi(self.sos) for _ in range(num_channels)])
        self.num_channels = num_channels

    def process_sample(self, sample: List[float]) -> List[float]:
        """Process a single sample of data (one value per channel)"""
        if len(sample) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {len(sample)}")
        
        # Convert to numpy array and reshape for processing
        x = np.array(sample)[:, np.newaxis]
        y = np.zeros_like(x)
        
        # Process each channel
        for ch in range(self.num_channels):
            y[ch], self.zi[ch] = sosfilt(self.sos, x[ch], zi=self.zi[ch])
            
        return y.flatten().tolist()

def run_benchmark(num_samples: int = 10000):
    """Run benchmark comparing stateful vs non-stateful filtering"""
    num_channels = 18
    sfreq = 500.0
    
    # Generate random test data
    data = np.random.randn(num_channels, num_samples)
    
    # Create stateful filter
    stateful_filter = StatefulSOSFilter(num_channels, sfreq)
    
    # Benchmark stateful filtering
    print("\nBenchmarking stateful filtering...")
    stateful_times = []
    filtered_stateful = []
    
    for i in range(num_samples):
        sample = data[:, i].tolist()
        start_time = time.perf_counter()
        filtered_sample = stateful_filter.process_sample(sample)
        end_time = time.perf_counter()
        stateful_times.append(end_time - start_time)
        filtered_stateful.append(filtered_sample)
    
    # Benchmark non-stateful filtering
    print("Benchmarking non-stateful filtering...")
    start_time = time.perf_counter()
    nyq = sfreq / 2
    low = 4.0 / nyq
    high = 40.0 / nyq
    sos = butter(4, [low, high], btype='band', output='sos')
    filtered_nonstateful = sosfilt(sos, data)
    end_time = time.perf_counter()
    total_time_nonstateful = end_time - start_time
    
    # Print results
    stateful_times = np.array(stateful_times)
    print("\nResults:")
    print(f"Stateful filtering:")
    print(f"  Average time per sample: {np.mean(stateful_times)*1e6:.2f} microseconds")
    print(f"  Min time per sample: {np.min(stateful_times)*1e6:.2f} microseconds")
    print(f"  Max time per sample: {np.max(stateful_times)*1e6:.2f} microseconds")
    print(f"  Total time: {np.sum(stateful_times):.4f} seconds")
    print(f"\nNon-stateful filtering:")
    print(f"  Total time: {total_time_nonstateful:.4f} seconds")
    print(f"  Average time per sample: {(total_time_nonstateful/num_samples)*1e6:.2f} microseconds")
    
    # Verify results match
    filtered_stateful = np.array(filtered_stateful).T
    max_diff = np.max(np.abs(filtered_stateful - filtered_nonstateful))
    print(f"\nMaximum difference between implementations: {max_diff:.2e}")

if __name__ == "__main__":
    run_benchmark()
