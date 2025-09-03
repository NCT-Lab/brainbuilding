import os
import numpy as np
import h5py
from pyriemann.utils.distance import distance_riemann
from tqdm import tqdm
import warnings
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.brainbuilding.training import compute_normalized_augmented_covariances
from brainbuilding.core.config import ORDER

# Configuration
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
INPUT_DATASET_FNAME = os.getenv("INPUT_DATASET_NAME", 'motor-imagery-2.h5')
OUTPUT_DATASET_FNAME = os.getenv("OUTPUT_DATASET_NAME", 'motor-imagery-distances-2.h5')

# Full paths
input_path = os.path.join(INPUT_DATA_DIR, INPUT_DATASET_FNAME)
output_path = os.path.join(INPUT_DATA_DIR, OUTPUT_DATASET_FNAME)

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from HDF5 file"""
    print(f"Loading data from: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Read the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # If the file exists but is in parquet format, inform the user
        if 'X' not in f:
            raise ValueError(f"Input file {file_path} does not contain expected datasets. "
                            "Please check if the file is in HDF5 format or needs conversion.")
        
        X = f['X'][:]
        y = f['y'][:]
        subject_ids = f['subject_ids'][:]
        sample_weights = f['sample_weights'][:]
        event_ids = f['event_ids'][:]
    
    print(f"Data loaded successfully:")
    print(f"X shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique subjects: {len(np.unique(subject_ids))}")
    
    return X, y, subject_ids, sample_weights, event_ids

def compute_distances(X, subject_ids):
    """Compute Riemannian distances between all samples within each subject"""
    print("\nComputing Riemannian distances within subjects...")
    
    # First compute covariances if not already computed
    X_cov = compute_normalized_augmented_covariances(X, order=1, lag=1)
    
    # Store distances for each subject
    unique_subjects = np.sort(np.unique(subject_ids))
    distances_dict = {}
    subject_indices = {}  # Store indices of samples for each subject
    
    for subject in tqdm(unique_subjects):
        # Get samples for this subject
        mask = (subject_ids == subject)
        samples = X_cov[mask]
        subject_indices[subject] = np.where(mask)[0]  # Store original indices
        
        # Skip if too few samples
        if len(samples) < 2:
            print(f"Skipping subject {subject}: too few samples ({len(samples)})")
            continue
        
        # Compute all pairwise distances
        n_samples = samples.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distances[i, j] = distance_riemann(samples[i], samples[j])
        
        # Mirror the matrix preserving the diagonal (already zeros)
        distances = distances + distances.T
        
        # Store the distances
        distances_dict[subject] = distances
    
    return distances_dict, X_cov, subject_indices

def save_distances(distances_dict, subject_indices, X_cov, X, y, subject_ids, sample_weights, event_ids, output_path):
    """Save distances to an HDF5 file"""
    print(f"\nSaving distances to: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save data to HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create datasets for original data
        f.create_dataset('X', data=X, compression='gzip')
        f.create_dataset('y', data=y, compression='gzip')
        f.create_dataset('subject_ids', data=subject_ids, compression='gzip')
        f.create_dataset('sample_weights', data=sample_weights, compression='gzip')
        f.create_dataset('event_ids', data=event_ids, compression='gzip')
        
        # Create dataset for covariance matrices
        f.create_dataset('X_cov', data=X_cov, compression='gzip')
        
        # Create a group for distances
        distances_group = f.create_group('distances')
        
        # Store each subject's distance matrix and indices
        for subject, matrix in distances_dict.items():
            subject_group = distances_group.create_group(str(subject))
            subject_group.create_dataset('matrix', data=matrix, compression='gzip')
            subject_group.create_dataset('indices', data=subject_indices[subject], compression='gzip')
    
    print("Data saved successfully:")
    print(f"Number of subjects with distance matrices: {len(distances_dict)}")
    print(f"Saved to: {output_path}")

def main():
    # Load data
    X, y, subject_ids, sample_weights, event_ids = load_data(input_path)
    
    # Compute Riemannian distances
    distances_dict, X_cov, subject_indices = compute_distances(X, subject_ids)
    
    # Save results
    save_distances(distances_dict, subject_indices, X_cov, X, y, subject_ids, sample_weights, event_ids, output_path)
    
    print("\nDistance computation completed successfully!")

if __name__ == "__main__":
    main()
