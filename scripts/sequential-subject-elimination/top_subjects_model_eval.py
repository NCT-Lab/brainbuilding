import os
import sys
import numpy as np
import h5py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pyriemann.spatialfilters import CSP
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from datetime import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.brainbuilding.training import AugmentedDataset
from brainbuilding.core.config import ORDER, LAG, CSP_METRIC

ORDER = 1
LAG = 1

# Configuration
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
INPUT_DATASET_FNAME = os.getenv("INPUT_DATASET_NAME", 'motor-imagery-2.h5')
KNN_METRICS_FILE = os.getenv("KNN_METRICS_FILE", 'scripts/sequential-subject-elimination/results/subject_knn_metrics_k5.csv')
NUM_TOP_SUBJECTS = 10  # Number of top subjects to use

# Full paths
input_path = os.path.join(INPUT_DATA_DIR, INPUT_DATASET_FNAME)
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
    
    print(f"Data loaded successfully:")
    print(f"X shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of unique subjects: {len(np.unique(subject_ids))}")
    
    return X, y, subject_ids, sample_weights, event_ids

def get_top_subjects(metrics_file, n=10, metric='f1'):
    """Get the top N subjects based on a specific metric"""
    print(f"Loading subject metrics from: {metrics_file}")
    
    # Check if the file exists
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    # Load metrics
    metrics_df = pd.read_csv(metrics_file)
    
    # Sort by the specified metric and get top N
    top_subjects = metrics_df.sort_values(metric, ascending=False).head(n)['subject_id'].tolist()
    
    print(f"Selected top {n} subjects based on {metric}:")
    for i, subject in enumerate(top_subjects):
        score = metrics_df[metrics_df['subject_id'] == subject][metric].values[0]
        print(f"{i+1}. Subject {subject}: {metric}={score:.4f}")
    
    return top_subjects

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
        Preprocessed features
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

def create_pipeline():
    """Create a pipeline with CSP and SVM"""
    steps = [
        ("csp", CSP(nfilter=10, metric=CSP_METRIC)), 
        ("svm", SVC(kernel='rbf', probability=True, C=0.696, gamma=0.035))
    ]
    
    return Pipeline(steps=steps)

def evaluate_model_with_cv(X_preprocessed, y, subject_ids, selected_subjects):
    """
    Evaluate the model using cross-validation on selected subjects
    
    Parameters:
    -----------
    X_preprocessed : array
        Preprocessed features
    y : array
        Labels
    subject_ids : array
        Subject IDs for each sample
    selected_subjects : list
        List of subject IDs to include
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Create mask for selected subjects
    mask = np.isin(subject_ids, selected_subjects)
    
    # Filter data to only include selected subjects
    X_subset = X_preprocessed[mask]
    y_subset = y[mask]
    subject_ids_subset = subject_ids[mask]
    
    # Create cross-validator where each fold is a subject
    cv = GroupKFold(n_splits=len(selected_subjects))
    
    # Store results for each fold
    fold_results = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X_subset, y_subset, subject_ids_subset)), 
                                               total=len(selected_subjects), desc="Cross-validation"):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y_subset[train_idx], y_subset[test_idx]
        
        # Get test subject ID for logging
        test_subject = np.unique(subject_ids_subset[test_idx])[0]
        
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
    
    # Calculate weighted mean for each metric based on number of samples
    weighted_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        weighted_metrics[f'weighted_{metric}'] = np.average(
            results_df[metric], weights=results_df['num_samples']
        )
        weighted_metrics[f'mean_{metric}'] = results_df[metric].mean()
    
    # Add aggregate metrics to results
    overall_results = {
        'fold_results': results_df,
        'metrics': weighted_metrics
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
    plt.title('Model Performance Metrics by Subject (Leave-One-Subject-Out CV)')
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
        ax.set_title(f'{metric.capitalize()} Score by Subject')
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
        metric_plot_path = os.path.join(os.path.dirname(output_path), f'top_subjects_model_{metric}.png')
        plt.savefig(metric_plot_path)
        print(f"{metric.capitalize()} plot saved to: {metric_plot_path}")
    
    return

def save_results(results, output_path):
    """Save detailed results to file"""
    # Create a copy of results with DataFrame converted to dict for serialization
    serializable_results = results.copy()
    serializable_results['fold_results'] = results['fold_results'].to_dict(orient='records')
    
    # Add timestamp
    serializable_results['timestamp'] = datetime.now().isoformat()
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(serializable_results, f)
    
    print(f"Results saved to: {output_path}")
    
    # Also save metrics as CSV for easy access
    csv_path = output_path.replace('.pkl', '.csv')
    results['fold_results'].to_csv(csv_path, index=False)
    print(f"Metrics CSV saved to: {csv_path}")

def main():
    # Get the top subjects based on F1 score from previous KNN evaluation
    top_subjects = get_top_subjects(KNN_METRICS_FILE, n=NUM_TOP_SUBJECTS, metric='f1')
    
    # Load the data
    X, y, subject_ids, sample_weights, event_ids = load_data(input_path)
    
    # Remove class 2 samples if needed
    mask = y != 2
    X = X[mask]
    y = y[mask]
    subject_ids = subject_ids[mask]
    
    # Preprocess data (apply augmentation and covariance estimation)
    X_preprocessed, y, subject_ids = preprocess_data(X, y, subject_ids)
    
    # Evaluate model with cross-validation on selected subjects
    results = evaluate_model_with_cv(X_preprocessed, y, subject_ids, top_subjects)
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot results
    plot_path = os.path.join(output_dir, f'top_subjects_model_results_{timestamp}.png')
    plot_fold_results(results['fold_results'], plot_path)
    
    # Save detailed results
    results_path = os.path.join(output_dir, f'top_subjects_model_results_{timestamp}.pkl')
    save_results(results, results_path)
    
    # Print overall results
    print("\nOverall Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 