import os
import numpy as np
import h5py
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
INPUT_DISTANCES_FNAME = os.getenv("INPUT_DISTANCES_NAME", 'motor-imagery-distances-2.h5')
OUTPUT_DIR = os.getenv("OUTPUT_DIR", 'scripts/sequential-subject-elimination/results')
NUM_NEIGHBORS = 5  # Number of neighbors for KNN

# Full paths
input_distances_path = os.path.join(INPUT_DATA_DIR, INPUT_DISTANCES_FNAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_distance_data(file_path):
    """Load distance matrices and associated data"""
    print(f"Loading distance data from: {file_path}")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Read the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Load original data
        X = f['X'][:]
        y = f['y'][:]
        subject_ids = f['subject_ids'][:]
        sample_weights = f['sample_weights'][:]
        event_ids = f['event_ids'][:]
        
        # Load covariance matrices if available
        X_cov = f['X_cov'][:] if 'X_cov' in f else None
        
        # Load distance matrices for each subject
        distances = {}
        subject_indices = {}
        
        for subject in f['distances']:
            distances[int(subject)] = f['distances'][subject]['matrix'][:]
            subject_indices[int(subject)] = f['distances'][subject]['indices'][:]
    
    print(f"Data loaded successfully:")
    print(f"X shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique subjects: {len(np.unique(subject_ids))}")
    print(f"Number of subjects with distance matrices: {len(distances)}")
    
    return X, y, subject_ids, sample_weights, event_ids, X_cov, distances, subject_indices

def knn_predict(distance_matrix, sample_indices, labels, event_ids, test_idx, k=5):
    """
    Predict class for test_idx using KNN based on distance matrix
    Excludes samples from the same event as the test sample
    
    Args:
        distance_matrix: matrix of distances between all samples
        sample_indices: original indices of samples in the distance matrix
        labels: labels for all samples
        event_ids: event ID for each sample
        test_idx: index in sample_indices of the test sample
        k: number of neighbors
        
    Returns:
        predicted class
    """
    # Get the test sample's event ID
    test_event_id = event_ids[sample_indices[test_idx]]
    
    # Get distances from test sample to all other samples
    distances = distance_matrix[test_idx, :]
    
    # Create mask to exclude samples from the same event
    mask = np.ones_like(distances, dtype=bool)
    mask[test_idx] = False  # Exclude the test sample itself
    
    # Exclude samples from the same event
    for i, idx in enumerate(sample_indices):
        if event_ids[idx] == test_event_id and i != test_idx:
            mask[i] = False
    
    # Apply mask to get valid neighbors
    valid_distances = distances[mask]
    valid_indices = np.arange(len(distances))[mask]
    
    # If no valid neighbors after filtering, use all except self
    if len(valid_distances) == 0:
        print(f"Warning: No valid neighbors found after filtering same event samples for test_idx {test_idx}")
        mask = np.ones_like(distances, dtype=bool)
        mask[test_idx] = False
        valid_distances = distances[mask]
        valid_indices = np.arange(len(distances))[mask]
    
    # If still not enough neighbors, use fewer neighbors
    k_actual = min(k, len(valid_distances))
    if k_actual < k:
        print(f"Warning: Only {k_actual} neighbors available, using all of them instead of requested {k}")
    
    # Get the k nearest neighbors
    nearest_indices = valid_indices[np.argsort(valid_distances)[:k_actual]]
    nearest_labels = [labels[sample_indices[i]] for i in nearest_indices]
    
    # Predict class (majority vote)
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

def evaluate_subject_knn(subject, distances, sample_indices, y, event_ids, global_indices):
    """
    Evaluate KNN performance for a single subject using leave-one-out cross-validation
    
    Args:
        subject: subject ID
        distances: distance matrix for the subject
        sample_indices: original indices of samples in the distance matrix
        y: labels for all samples
        event_ids: event ID for each sample
        global_indices: mapping from local to global indices
        
    Returns:
        dictionary with evaluation metrics
    """
    n_samples = len(sample_indices)
    y_true = []
    y_pred = []
    
    for i in tqdm(range(n_samples), desc=f"Subject {subject}", leave=False):
        true_label = y[sample_indices[i]]
        pred_label = knn_predict(distances, sample_indices, y, event_ids, i, k=NUM_NEIGHBORS)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    # Calculate metrics
    metrics = {
        'subject_id': subject,
        'num_samples': n_samples,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics, y_true, y_pred

def create_metrics_plots(metrics_df, output_dir):
    """Create plots of evaluation metrics for all subjects"""
    print("\nCreating metrics plots...")
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 10))
    
    # Plot all metrics on one plot
    ax = plt.subplot(111)
    
    # Get metrics to plot (excluding non-metric columns)
    metric_cols = ['accuracy', 'precision', 'recall', 'f1']
    
    # Sort subjects by F1 score
    metrics_df = metrics_df.sort_values('f1', ascending=False)
    
    # Create bar plot
    x = np.arange(len(metrics_df))
    width = 0.2
    offsets = [-width*1.5, -width/2, width/2, width*1.5]
    
    for i, metric in enumerate(metric_cols):
        ax.bar(x + offsets[i], metrics_df[metric], width, label=metric.capitalize())
    
    # Add labels and legend
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Score')
    ax.set_title('KNN Performance Metrics by Subject (k={})'.format(NUM_NEIGHBORS))
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['subject_id'])
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'subject_knn_metrics_k{NUM_NEIGHBORS}.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Create additional plots for each metric separately
    for metric in metric_cols:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='subject_id', y=metric, data=metrics_df.sort_values(metric, ascending=False))
        ax.set_title(f'{metric.capitalize()} Score by Subject (k={NUM_NEIGHBORS})')
        ax.set_xlabel('Subject ID')
        ax.set_ylabel(metric.capitalize())
        
        # Add value labels on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points')
        
        plt.tight_layout()
        metric_plot_path = os.path.join(output_dir, f'subject_knn_{metric}_k{NUM_NEIGHBORS}.png')
        plt.savefig(metric_plot_path)
        print(f"{metric.capitalize()} plot saved to: {metric_plot_path}")

def main():
    # Load distance data
    X, y, subject_ids, sample_weights, event_ids, X_cov, distances, subject_indices = load_distance_data(input_distances_path)
    
    # Store metrics for all subjects
    all_metrics = []
    
    # Evaluate each subject
    print("\nEvaluating KNN for each subject...")
    
    for subject in tqdm(distances.keys(), desc="Processing subjects"):
        # Get distance matrix and sample indices for this subject
        subject_distance_matrix = distances[subject]
        subject_sample_indices = subject_indices[subject]
        
        # Create mapping from local to global indices
        global_indices = {i: idx for i, idx in enumerate(subject_sample_indices)}
        
        # Skip subjects with too few samples
        if len(subject_sample_indices) < NUM_NEIGHBORS + 1:
            print(f"Skipping subject {subject}: too few samples ({len(subject_sample_indices)})")
            continue
        
        # Evaluate KNN for this subject
        metrics, subject_y_true, subject_y_pred = evaluate_subject_knn(
            subject, subject_distance_matrix, subject_sample_indices, y, event_ids, global_indices
        )
        
        all_metrics.append(metrics)
        
        # Print metrics for this subject
        print(f"\nSubject {subject} metrics:")
        for key, value in metrics.items():
            if key != 'subject_id':
                print(f"  {key}: {value:.4f}")
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    csv_path = os.path.join(OUTPUT_DIR, f'subject_knn_metrics_k{NUM_NEIGHBORS}.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")
    
    # Create plots
    create_metrics_plots(metrics_df, OUTPUT_DIR)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
