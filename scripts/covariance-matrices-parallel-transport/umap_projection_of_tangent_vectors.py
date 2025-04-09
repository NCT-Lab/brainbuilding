import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from datetime import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
TANGENT_DATASET_FNAME = os.getenv("TANGENT_DATASET_NAME", 'motor-imagery-tangent-space.h5')

# Full paths
input_path = os.path.join(INPUT_DATA_DIR, TANGENT_DATASET_FNAME)
output_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    """Load tangent space vectors from HDF5 file"""
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
    
    print(f"Data loaded successfully:")
    print(f"X shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique subjects: {len(np.unique(subject_ids))}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y, subject_ids

def project_to_2d(X, y, n_neighbors=None):
    """
    Project tangent vectors to 2D using UMAP
    
    Parameters:
    -----------
    X : array
        Tangent vectors
    y : array
        Labels
    n_neighbors : int
        Number of neighbors for UMAP (if None, calculated based on samples/classes)
    
    Returns:
    --------
    array
        2D projections
    """
    # Calculate number of neighbors if not provided
    if n_neighbors is None:
        n_classes = len(np.unique(y))
        n_samples = len(X)
        n_neighbors = max(2, int(n_samples / n_classes))
    
    print(f"Projecting to 2D with UMAP using {n_neighbors} neighbors...")
    
    # Configure UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='euclidean',
    )
    
    # Fit and transform
    X_2d = reducer.fit_transform(X)
    
    print(f"Projection complete. Shape: {X_2d.shape}")
    
    return X_2d

def plot_projections(X_2d, y, subject_ids, output_dir):
    """
    Plot 2D projections with different colorings
    
    Parameters:
    -----------
    X_2d : array
        2D projections
    y : array
        Labels
    subject_ids : array
        Subject IDs
    output_dir : str
        Directory to save plots
    """
    
    # Plot colored by class
    plt.figure(figsize=(12, 10))
    unique_classes = np.unique(y)
    class_names = [f"Class {c}" for c in unique_classes]
    
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.7, s=10)
    
    plt.colorbar(ticks=unique_classes, label='Class')
    plt.title('UMAP Projection of Tangent Vectors (Colored by Class)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    
    # # Add legend
    # for i, cls in enumerate(unique_classes):
    #     plt.scatter([], [], c=[i], cmap='tab10', label=class_names[i])
    plt.legend()
    
    plt.tight_layout()
    class_plot_path = os.path.join(output_dir, 'umap_projection_by_class.png')
    plt.savefig(class_plot_path, dpi=300)
    print(f"Class plot saved to: {class_plot_path}")
    
    # Plot colored by subject
    plt.figure(figsize=(14, 12))
    unique_subjects = np.unique(subject_ids)
    
    # Create a colormap with enough colors
    cmap = plt.cm.get_cmap('tab20', len(unique_subjects))
    subject_colors = {subj: cmap(i) for i, subj in enumerate(unique_subjects)}
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                    color=subject_colors[subj], 
                    alpha=0.7, 
                    s=10, 
                    label=f"Subject {subj}")
    
    plt.title('UMAP Projection of Tangent Vectors (Colored by Subject)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    
    # Add legend with decent size
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    subject_plot_path = os.path.join(output_dir, 'umap_projection_by_subject.png')
    plt.savefig(subject_plot_path, dpi=300, bbox_inches='tight')
    print(f"Subject plot saved to: {subject_plot_path}")
    
    # Create class-subject combined plot
    plt.figure(figsize=(16, 14))
    
    # Create a scatter plot for each subject and class combination
    for subj in unique_subjects:
        for cls in unique_classes:
            mask = (subject_ids == subj) & (y == cls)
            if np.any(mask):  # Only plot if there are data points
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           alpha=0.7, 
                           s=10, 
                           label=f"Subject {subj} - Class {cls}")
    
    plt.title('UMAP Projection of Tangent Vectors (By Subject and Class)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    
    # Add legend with decent size
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, 'umap_projection_combined.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {combined_plot_path}")
    
    # Save the 2D projections and metadata as a CSV for further analysis
    df = pd.DataFrame({
        'umap_1': X_2d[:, 0],
        'umap_2': X_2d[:, 1],
        'class': y,
        'subject_id': subject_ids
    })
    
    csv_path = os.path.join(output_dir, 'umap_projection_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Projection data saved to: {csv_path}")

def main():
    # Load the data
    X, y, subject_ids = load_data(input_path)
    
    # Calculate n_neighbors based on samples/classes
    n_classes = len(np.unique(y))
    n_samples = len(X)
    n_neighbors = max(2, int(n_samples / n_classes))
    
    print(f"Using {n_neighbors} neighbors for UMAP projection")
    
    # Project to 2D
    X_2d = project_to_2d(X, y, n_neighbors=n_neighbors)
    
    # Plot projections
    plot_projections(X_2d, y, subject_ids, output_dir)

if __name__ == "__main__":
    main()
