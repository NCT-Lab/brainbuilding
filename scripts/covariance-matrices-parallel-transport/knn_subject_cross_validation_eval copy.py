import os
import sys
import numpy as np
import h5py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from datetime import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
TANGENT_DATASET_FNAME = os.getenv("TANGENT_DATASET_NAME", 'motor-imagery-tangent-space.h5')
KNN_NEIGHBORS = os.getenv("KNN_NEIGHBORS", 25)  # Number of neighbors for KNN
# EXCLUDE_SUBJECTS = [6, 14, 9, 3]  # Subjects to exclude from analysis
EXCLUDE_SUBJECTS = []

# Full paths
input_path = os.path.join(INPUT_DATA_DIR, TANGENT_DATASET_FNAME)
output_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(output_dir, exist_ok=True)

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
    
    # Exclude specified subjects
    mask = ~np.isin(subject_ids, EXCLUDE_SUBJECTS)
    X = X[mask]
    y = y[mask]
    subject_ids = subject_ids[mask]
    sample_weights = sample_weights[mask]
    event_ids = event_ids[mask]
    
    print(f"Data loaded successfully:")
    print(f"X shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique subjects: {len(np.unique(subject_ids))}")
    print(f"Excluded subjects: {EXCLUDE_SUBJECTS}")
    
    return X, y, subject_ids, sample_weights, event_ids

def create_pipeline():
    """Create a pipeline with KNN classifier"""
    steps = [
        ("knn", KNeighborsClassifier(n_neighbors=int(KNN_NEIGHBORS), weights='distance'))
    ]
    
    return Pipeline(steps=steps)

def evaluate_model_with_cv(X, y, subject_ids):
    """
    Evaluate the model using leave-one-subject-out cross-validation
    
    Parameters:
    -----------
    X : array
        Tangent vectors features
    y : array
        Labels
    subject_ids : array
        Subject IDs for each sample
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Get all unique subject IDs
    unique_subjects = np.unique(subject_ids)
    
    # Create cross-validator where each fold is a subject
    cv = GroupKFold(n_splits=len(unique_subjects))
    
    # Store results for each fold
    fold_results = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X, y, subject_ids)), 
                                              total=len(unique_subjects), desc="Cross-validation"):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get test subject ID for logging
        test_subject = np.unique(subject_ids[test_idx])[0]
        
        # Create and train the pipeline
        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Predict on test fold
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        fold_metrics = {
            'subject_id': test_subject,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'num_samples': len(y_test)
        }
        
        fold_results.append(fold_metrics)
        
        # Print fold results
        print(f"\nFold {fold_idx+1} (Subject {test_subject}):")
        for metric, value in fold_metrics.items():
            if metric not in ['subject_id', 'num_samples']:
                print(f"  {metric}: {value:.4f}")
    
    # Convert to DataFrame and calculate aggregate metrics
    results_df = pd.DataFrame(fold_results)
    
    # Calculate weighted metrics and standard deviations
    metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        # Calculate weighted mean
        metrics[metric] = np.average(
            results_df[metric], weights=results_df['num_samples']
        )
        # Calculate weighted standard deviation
        metrics[f'{metric}_std'] = np.sqrt(
            np.average(
                (results_df[metric] - metrics[metric])**2,
                weights=results_df['num_samples']
            )
        )
    
    # Add aggregate metrics to results
    overall_results = {
        'fold_results': results_df,
        'metrics': metrics
    }
    
    return overall_results

def plot_fold_results(results_df, output_path):
    """Plot metrics for each fold/subject"""
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot metrics
    metric_cols = ['accuracy', 'precision', 'recall', 'f1']
    
    # Sort by F1 score
    results_df = results_df.sort_values('f1', ascending=False)
    
    # Create bar plot
    x = np.arange(len(results_df))
    width = 0.2
    offsets = [-width*1.5, -width/2, width/2, width*1.5]
    
    for i, metric in enumerate(metric_cols):
        plt.bar(x + offsets[i], results_df[metric], width, label=metric.capitalize())
    
    # Add labels and legend
    plt.xlabel('Subject ID')
    plt.ylabel('Score')
    plt.title('KNN Performance Metrics by Subject (Leave-One-Subject-Out CV)')
    plt.xticks(x, results_df['subject_id'])
    plt.legend()
    
    # Add mean lines
    for i, metric in enumerate(metric_cols):
        mean_value = results_df[metric].mean()
        plt.axhline(mean_value, color=f'C{i}', linestyle='--', alpha=0.7,
                   label=f'Mean {metric.capitalize()}: {mean_value:.4f}')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    
    # Create additional plots for each metric separately
    for metric in metric_cols:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='subject_id', y=metric, data=results_df.sort_values(metric, ascending=False))
        ax.set_title(f'{metric.capitalize()} Score by Subject (KNN)')
        ax.set_xlabel('Subject ID')
        ax.set_ylabel(metric.capitalize())
        
        # Add mean line
        mean_value = results_df[metric].mean()
        plt.axhline(mean_value, color='r', linestyle='--', 
                   label=f'Mean: {mean_value:.4f}')
        plt.legend()
        
        # Add value labels on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.3f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', 
                       xytext=(0, 10), 
                       textcoords='offset points')
        
        plt.tight_layout()
        metric_plot_path = os.path.join(os.path.dirname(output_path), f'knn_subject_cv_{metric}.png')
        plt.savefig(metric_plot_path)
        print(f"{metric.capitalize()} plot saved to: {metric_plot_path}")
    
    return

def save_results(results, output_path):
    """Save detailed results to file"""
    # Create a copy of results with DataFrame converted to dict for serialization
    serializable_results = results.copy()
    serializable_results['fold_results'] = results['fold_results'].to_dict(orient='records')
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(serializable_results, f)
    
    print(f"Results saved to: {output_path}")
    
    # Also save metrics as CSV for easy access
    csv_path = output_path.replace('.pkl', '.csv')
    results['fold_results'].to_csv(csv_path, index=False)
    print(f"Metrics CSV saved to: {csv_path}")

def main():
    # Load the tangent space data
    X, y, subject_ids, sample_weights, event_ids = load_data(input_path)
    
    # Remove class 2 samples if needed
    mask = y != 2
    X = X[mask]
    y = y[mask]
    subject_ids = subject_ids[mask]
    
    # Evaluate model with cross-validation on all subjects
    results = evaluate_model_with_cv(X, y, subject_ids)
    
    # Plot results
    plot_path = os.path.join(output_dir, f'knn_subject_cv_results.png')
    plot_fold_results(results['fold_results'], plot_path)
    
    # Save detailed results
    results_path = os.path.join(output_dir, f'knn_subject_cv_results.pkl')
    save_results(results, results_path)
    
    # Print overall results
    print(f"Excluded subjects: {EXCLUDE_SUBJECTS}")
    print("\nOverall Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
