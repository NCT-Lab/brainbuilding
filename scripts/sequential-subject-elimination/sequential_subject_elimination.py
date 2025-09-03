import os
import sys
import numpy as np
import h5py
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from datetime import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.brainbuilding.training import AugmentedDataset
from brainbuilding.core.config import ORDER, LAG, CSP_METRIC

# Configuration
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
INPUT_DATASET_FNAME = os.getenv("INPUT_DATASET_NAME", 'motor-imagery-2.h5')

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
    covariances = Covariances(estimator="oas")
    
    # Apply augmentation
    print("Applying augmentation...")
    X_aug = augmentation.transform(X)
    
    # Apply covariance estimation
    print("Computing covariances...")
    X_cov = covariances.transform(X_aug)
    
    print(f"Preprocessing complete. New feature shape: {X_cov.shape}")
    
    return X_cov, y, subject_ids

def create_simplified_pipeline():
    """Create a pipeline that works with pre-processed features"""
    steps = [
        ("csp", CSP(nfilter=10, metric=CSP_METRIC)), 
        ("svm", SVC(kernel='rbf', probability=True, C=0.696, gamma=0.035))
    ]
    
    return Pipeline(steps=steps)

def evaluate_subject_set(X_preprocessed, y, included_subjects, subject_ids):
    """
    Evaluate a set of subjects using cross-validation
    
    Parameters:
    -----------
    X_preprocessed : array
        Preprocessed features (after augmentation and covariance estimation)
    y : array
        Labels
    included_subjects : list
        List of subject IDs to include
    subject_ids : array
        Array of subject IDs for each sample
    
    Returns:
    --------
    float
        Mean F1 score across folds
    dict
        Dictionary with detailed scores per fold
    """
    # Create mask for included subjects
    mask = np.isin(subject_ids, included_subjects)
    
    # Filter data
    X_subset = X_preprocessed[mask]
    y_subset = y[mask]
    subject_ids_subset = subject_ids[mask]
    
    # Create cross-validator where each fold is a subject
    cv = GroupKFold(n_splits=len(included_subjects))
    
    # Store scores for each fold
    fold_scores = []
    fold_details = {}
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in tqdm(enumerate(cv.split(X_subset, y_subset, subject_ids_subset)), total=len(included_subjects)):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y_subset[train_idx], y_subset[test_idx]
        
        # Get test subject ID for logging
        test_subject = np.unique(subject_ids_subset[test_idx])[0]
        
        # Create and train the pipeline
        pipeline = create_simplified_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Predict on test fold
        y_pred = pipeline.predict(X_test)
        
        # Calculate F1 score
        score = f1_score(y_test, y_pred)
        fold_scores.append(score)
        
        # Store details for this fold
        fold_details[test_subject] = {
            'f1_score': score,
            'num_samples': len(y_test)
        }
        
    return np.mean(fold_scores), fold_details

def sequential_subject_elimination(X_preprocessed, y, subject_ids, num_subjects_to_keep=10):
    """
    Sequentially eliminate subjects to maximize F1 score
    
    Parameters:
    -----------
    X_preprocessed : array
        Preprocessed features
    y : array
        Labels
    subject_ids : array
        Array of subject IDs for each sample
    num_subjects_to_keep : int
        Number of subjects to keep
    
    Returns:
    --------
    list
        List of subject IDs to keep
    list
        List of subjects eliminated in order
    list
        F1 scores at each elimination step
    """
    # Get unique subject IDs
    all_subjects = np.unique(subject_ids)
    num_subjects = len(all_subjects)
    
    if num_subjects <= num_subjects_to_keep:
        print(f"Only {num_subjects} subjects available, fewer than requested {num_subjects_to_keep} to keep.")
        return all_subjects, [], []
    
    # Initialize with all subjects
    current_subjects = list(all_subjects)
    eliminated_subjects = []
    f1_scores = []
    
    # Initial evaluation with all subjects
    initial_f1, _ = evaluate_subject_set(X_preprocessed, y, current_subjects, subject_ids)
    print(f"Initial F1 score with all {len(current_subjects)} subjects: {initial_f1:.4f}")
    
    # Sequentially eliminate subjects until we reach the desired number
    with tqdm(total=num_subjects - num_subjects_to_keep, desc="Eliminating subjects") as pbar:
        while len(current_subjects) > num_subjects_to_keep:
            worst_f1 = float('inf')  # Start with infinity for finding minimum
            subject_to_eliminate = None
            best_overall_f1 = -1  # For tracking the best overall model
            details_for_best_overall = None
            
            # Try removing each subject and keep track of individual subject performance
            subject_performances = {}
            
            # First, evaluate each subject individually to get baseline performance
            print("\nEvaluating individual subject performances...")
            for subject in current_subjects:
                # Create a candidate set without the current subject
                candidate_subjects = [s for s in current_subjects if s != subject]
                
                # Evaluate the candidate set
                f1, details = evaluate_subject_set(X_preprocessed, y, candidate_subjects, subject_ids)
                
                # Store this subject's performance impact
                subject_performances[subject] = f1
                
                # Track the best overall model configuration
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    details_for_best_overall = details
                
                # If removing this subject gives the lowest F1, it's the worst contributor
                if f1 < worst_f1:
                    worst_f1 = f1
                    subject_to_eliminate = subject
            
            # Print individual performance impacts
            print("\nPerformance impact when each subject is removed:")
            for subject, perf in sorted(subject_performances.items(), key=lambda x: x[1]):
                impact = perf - initial_f1
                print(f"  Subject {subject}: F1={perf:.4f} (impact: {impact:+.4f})")
            
            # Eliminate the subject that contributes least (removing it causes worst performance)
            subject_to_eliminate = max(subject_performances, key=subject_performances.get)
            current_subjects.remove(subject_to_eliminate)
            eliminated_subjects.append(subject_to_eliminate)
            f1_scores.append(best_overall_f1)
            
            # Update initial f1 for next iteration
            initial_f1 = best_overall_f1
            
            print(f"\nRemoved subject {subject_to_eliminate}, new F1 score: {best_overall_f1:.4f}")
            print(f"Detailed scores for remaining subjects:")
            for subj, detail in details_for_best_overall.items():
                print(f"  Subject {subj}: F1={detail['f1_score']:.4f} (samples: {detail['num_samples']})")
            
            pbar.update(1)
            
    return current_subjects, eliminated_subjects, f1_scores

def plot_elimination_results(eliminated_subjects, f1_scores, output_path):
    """Plot F1 scores as subjects are eliminated"""
    plt.figure(figsize=(12, 6))
    
    # Create x-axis labels (number of subjects remaining)
    num_subjects_initial = len(eliminated_subjects) + 10  # As we keep 10 subjects
    x_values = list(range(num_subjects_initial - 1, 9, -1))
    
    plt.plot(x_values, f1_scores, 'o-', linewidth=2)
    plt.axhline(f1_scores[0], color='r', linestyle='--', label=f'Initial F1: {f1_scores[0]:.4f}')
    plt.axhline(f1_scores[-1], color='g', linestyle='--', label=f'Final F1: {f1_scores[-1]:.4f}')
    
    plt.xlabel('Number of Subjects Remaining')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Improvement through Sequential Subject Elimination')
    plt.grid(True)
    plt.legend(loc='best')
    
    # Add eliminated subject IDs as annotations
    for i, subj in enumerate(eliminated_subjects):
        plt.annotate(f'Subj {subj}', (x_values[i], f1_scores[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def save_results(kept_subjects, eliminated_subjects, f1_scores, output_path):
    """Save results as a pickle file"""
    results = {
        'kept_subjects': kept_subjects,
        'eliminated_subjects': eliminated_subjects,
        'f1_scores': f1_scores,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_path}")

def main():
    # Load the data
    X, y, subject_ids, sample_weights, event_ids = load_data(input_path)
    
    # Remove class 2 samples (as in training.py)
    mask = y != 2
    X = X[mask]
    y = y[mask]
    subject_ids = subject_ids[mask]
    
    # Preprocess data once (apply augmentation and covariance estimation)
    X_preprocessed, y, subject_ids = preprocess_data(X, y, subject_ids)
    
    # Run sequential subject elimination with preprocessed data
    kept_subjects, eliminated_subjects, f1_scores = sequential_subject_elimination(
        X_preprocessed, y, subject_ids, num_subjects_to_keep=10
    )
    
    # Create a timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot the results
    plot_path = os.path.join(output_dir, f'elimination_results_{timestamp}.png')
    plot_elimination_results(eliminated_subjects, f1_scores, plot_path)
    
    # Save detailed results
    results_path = os.path.join(output_dir, f'elimination_results_{timestamp}.pkl')
    save_results(kept_subjects, eliminated_subjects, f1_scores, results_path)
    
    # Print final subjects to keep
    print("\nSubjects to keep:")
    print(sorted(kept_subjects))
    
    print("\nSubjects eliminated (in order):")
    print(eliminated_subjects)
    
    print(f"\nInitial F1 score: {f1_scores[0]:.4f}")
    print(f"Final F1 score: {f1_scores[-1]:.4f}")
    print(f"Improvement: {(f1_scores[-1] - f1_scores[0]) * 100:.2f}%")

if __name__ == "__main__":
    main()
