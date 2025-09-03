import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
import warnings
from scipy.linalg import sqrtm
from pyriemann.utils.tangentspace import tangent_space

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.brainbuilding.training import AugmentedDataset
from brainbuilding.core.config import ORDER, LAG

ORDER = 1
LAG = 1

# Configuration
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
INPUT_DATASET_FNAME = os.getenv("INPUT_DATASET_NAME", 'motor-imagery-2.h5')
OUTPUT_COV_DATASET_FNAME = os.getenv("OUTPUT_COV_DATASET_NAME", 'motor-imagery-parallel-transport.h5')
OUTPUT_TANGENT_DATASET_FNAME = os.getenv("OUTPUT_TANGENT_DATASET_NAME", 'motor-imagery-tangent-space.h5')

# Full paths
input_path = os.path.join(INPUT_DATA_DIR, INPUT_DATASET_FNAME)
output_cov_path = os.path.join(INPUT_DATA_DIR, OUTPUT_COV_DATASET_FNAME)
output_tangent_path = os.path.join(INPUT_DATA_DIR, OUTPUT_TANGENT_DATASET_FNAME)

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from HDF5 file"""
    print(f"Loading data from: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Read the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # If the file exists but doesn't contain expected datasets
        if 'X' not in f:
            raise ValueError(f"Input file {file_path} does not contain expected datasets.")
        
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

def preprocess_data(X, y, subject_ids):
    """
    Preprocess the data using augmentation and covariance estimation
    
    Parameters:
    -----------
    X : array
        Raw features
    y : array
        Labels
    subject_ids : array
        Subject IDs for each sample
    
    Returns:
    --------
    array
        Preprocessed features (covariance matrices)
    array
        Labels (unchanged)
    array
        Subject IDs (unchanged)
    """
    print("Preprocessing data (augmentation and covariance estimation)...")
    
    # Create and apply the transformations
    augmentation = AugmentedDataset(order=ORDER, lag=LAG)
    
    # Apply augmentation
    print("Applying augmentation...")
    X_aug = augmentation.transform(X)
    
    # Apply covariance estimation
    print("Computing covariances...")
    from pyriemann.estimation import Covariances
    X_cov = Covariances(estimator="oas").transform(X_aug)
    
    print(f"Preprocessing complete. New feature shape: {X_cov.shape}")
    
    return X_cov, y, subject_ids

def compute_subject_mean_matrices(X_cov, subject_ids):
    """
    Compute mean covariance matrix for each subject using Riemannian mean
    
    Parameters:
    -----------
    X_cov : array
        Covariance matrices
    subject_ids : array
        Subject IDs for each sample
    
    Returns:
    --------
    dict
        Dictionary with subject IDs as keys and mean matrices as values
    array
        General mean matrix across all subjects
    """
    from pyriemann.utils.mean import mean_riemann
    
    # Get unique subject IDs
    unique_subjects = np.unique(subject_ids)
    print(f"Computing mean matrices for {len(unique_subjects)} subjects...")
    
    # Compute mean matrix for each subject
    subject_means = {}
    for subject in tqdm(unique_subjects, desc="Computing subject means"):
        mask = subject_ids == subject
        subject_matrices = X_cov[mask]
        subject_means[subject] = mean_riemann(subject_matrices)
    
    # Compute general mean across all subjects
    print("Computing general mean matrix...")
    general_mean = mean_riemann(X_cov)
    
    return subject_means, general_mean

def apply_parallel_transport(X_cov, subject_ids, subject_means, general_mean):
    """
    Apply parallel transport to covariance matrices
    
    Parameters:
    -----------
    X_cov : array
        Covariance matrices
    subject_ids : array
        Subject IDs for each sample
    subject_means : dict
        Dictionary with subject IDs as keys and mean matrices as values
    general_mean : array
        General mean matrix across all subjects
    
    Returns:
    --------
    array
        Transformed covariance matrices
    """
    print("Applying parallel transport to covariance matrices...")
    
    # Initialize array for transformed matrices
    X_transformed = np.zeros_like(X_cov)
    
    # Compute transformation matrices for each subject
    transform_matrices = {}
    for subject, M in subject_means.items():
        # Compute E = (GM^-1)^1/2
        GM_inv = np.dot(general_mean, np.linalg.inv(M))
        # GM_inv = M

        transform_matrices[subject] = sqrtm(GM_inv)
    
    # Apply transformation to each sample
    for i, (cov, subject) in enumerate(tqdm(zip(X_cov, subject_ids), 
                                           total=len(X_cov), 
                                           desc="Applying parallel transport")):
        E = transform_matrices[subject]
        # Compute ESE^T
        X_transformed[i] = np.dot(np.dot(E, cov), E.T)
    
    return X_transformed

def save_datasets(X_transformed, X_tangent, y, subject_ids, sample_weights, event_ids):
    """
    Save transformed datasets to HDF5 files
    
    Parameters:
    -----------
    X_transformed : array
        Transformed covariance matrices
    X_tangent : array
        Tangent space vectors
    y : array
        Labels
    subject_ids : array
        Subject IDs
    sample_weights : array
        Sample weights
    event_ids : array
        Event IDs
    """
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_cov_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_tangent_path), exist_ok=True)
    
    # Save transformed covariance matrices
    print(f"Saving transformed covariance matrices to: {output_cov_path}")
    with h5py.File(output_cov_path, 'w') as f:
        f.create_dataset('X', data=X_transformed, compression='gzip')
        f.create_dataset('y', data=y, compression='gzip')
        f.create_dataset('subject_ids', data=subject_ids, compression='gzip')
        f.create_dataset('sample_weights', data=sample_weights, compression='gzip')
        f.create_dataset('event_ids', data=event_ids, compression='gzip')
    
    # Save tangent space vectors
    print(f"Saving tangent space vectors to: {output_tangent_path}")
    with h5py.File(output_tangent_path, 'w') as f:
        f.create_dataset('X', data=X_tangent, compression='gzip')
        f.create_dataset('y', data=y, compression='gzip')
        f.create_dataset('subject_ids', data=subject_ids, compression='gzip')
        f.create_dataset('sample_weights', data=sample_weights, compression='gzip')
        f.create_dataset('event_ids', data=event_ids, compression='gzip')
    
    print("Datasets saved successfully!")

def main():
    # Load the data
    X, y, subject_ids, sample_weights, event_ids = load_data(input_path)
    
    # Remove class 2 samples if needed
    # mask = (y != 2) & (subject_ids != 23) & (subject_ids != 24)
    mask = (y != 2)
    X = X[mask]
    y = y[mask]
    subject_ids = subject_ids[mask]
    sample_weights = sample_weights[mask]
    event_ids = event_ids[mask]
    
    # Preprocess data (compute covariance matrices)
    X_cov, y, subject_ids = preprocess_data(X, y, subject_ids)
    
    # Compute subject mean matrices and general mean
    subject_means, general_mean = compute_subject_mean_matrices(X_cov, subject_ids)
    
    # Apply parallel transport to covariance matrices
    X_transformed = apply_parallel_transport(X_cov, subject_ids, subject_means, general_mean)
    
    # Project transformed matrices to tangent space
    print("Projecting matrices to tangent space...")
    X_tangent = tangent_space(X_transformed, general_mean)
    # X_tangent = []
    # for i, (cov, subject) in enumerate(tqdm(zip(X_transformed, subject_ids), 
    #                                        total=len(X_transformed), 
    #                                        desc="Projecting to tangent space")):
    #     X_tangent.append(tangent_space(cov, subject_means[subject]))
    # X_tangent = np.array(X_tangent)
    print(f"Tangent space projection complete. New feature shape: {X_tangent.shape}")
    
    # Save datasets
    save_datasets(X_transformed, X_tangent, y, subject_ids, sample_weights, event_ids)

if __name__ == "__main__":
    main()
