"""
Compare evaluation results between standard preprocessing and online preprocessing
using the hybrid pipeline evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.brainbuilding.pipelines import (
    WHITENING, BACKGROUND_FILTER, AUGMENTED_OAS_COV,
    PT_CSP_LOG, SIMPLE_CSP, SVC_CLASSIFICATION
)
from src.brainbuilding.evaluation import evaluate_pipeline

# --- Configuration ---
STANDARD_DATASET = 'data/preprocessed/motor-imagery-2.npy'
ONLINE_DATASET = 'data/preprocessed/motor-imagery-online.npy'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    """Load data from the NPY file."""
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    X = data
    y = data['label']
    print(f"X shape: {X['sample'].shape}")
    print(f"Number of subjects: {len(np.unique(X['subject_id']))}")
    print(f"Unique labels: {np.unique(y)}")

    print("Data loaded successfully.")
    return X, y

def create_pt_pipeline_steps():
    """Creates the pipeline steps with Parallel Transport."""
    return WHITENING + BACKGROUND_FILTER + AUGMENTED_OAS_COV + PT_CSP_LOG + SVC_CLASSIFICATION

def create_no_pt_pipeline_steps():
    """Creates the pipeline steps without Parallel Transport."""
    return WHITENING + BACKGROUND_FILTER + AUGMENTED_OAS_COV + SIMPLE_CSP + SVC_CLASSIFICATION

def run_hybrid_evaluation(X, y, dataset_name=""):
    """
    Evaluate both pipelines and collect their predictions.
    """
    print(f"Running hybrid evaluation for {dataset_name}...")
    
    # Evaluate PT pipeline and get predictions
    print("Evaluating PT pipeline...")
    pt_predictions = evaluate_pipeline(X, y, create_pt_pipeline_steps(), return_predictions=True)
    pt_predictions = pt_predictions.rename(columns={'predicted_label': 'pred_pt'})
    
    # Evaluate non-PT pipeline and get predictions
    print("Evaluating No-PT pipeline...")
    no_pt_predictions = evaluate_pipeline(X, y, create_no_pt_pipeline_steps(), return_predictions=True)
    no_pt_predictions = no_pt_predictions.rename(columns={'predicted_label': 'pred_no_pt'})
    
    # Merge predictions on subject_id and sample_index
    merged_predictions = pd.merge(
        no_pt_predictions[['subject_id', 'sample_index', 'true_label', 'pred_no_pt']],
        pt_predictions[['subject_id', 'sample_index', 'pred_pt']],
        on=['subject_id', 'sample_index']
    )
    
    return merged_predictions

def find_optimal_n(predictions_df):
    """
    Find the optimal N for switching from no-PT to PT pipeline.
    """
    print("Finding optimal N...")
    
    subjects = predictions_df['subject_id'].unique()
    max_samples = predictions_df.groupby('subject_id').size().max()
    
    best_n = -1
    best_f1 = -1
    
    f1_scores_by_n = []

    for n in range(max_samples + 1):
        hybrid_preds = []
        true_labels = []

        for subject_id in subjects:
            subject_df = predictions_df[predictions_df['subject_id'] == subject_id].sort_values('sample_index')
            
            # Non-PT predictions for the first N samples
            no_pt_preds = subject_df['pred_no_pt'].iloc[:n]
            
            # PT predictions for the remaining samples
            pt_preds = subject_df['pred_pt'].iloc[n:]
            
            # Combine to form hybrid predictions for the subject
            subject_hybrid_preds = pd.concat([no_pt_preds, pt_preds])
            
            hybrid_preds.extend(subject_hybrid_preds)
            true_labels.extend(subject_df['true_label'])
        
        # Calculate overall F1 for this N
        current_f1 = f1_score(true_labels, hybrid_preds, average='weighted')
        f1_scores_by_n.append({'n': n, 'f1_score': current_f1})
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_n = n
            
    print(f"Optimal N found: {best_n} with F1-score: {best_f1:.4f}")
    
    return best_n, pd.DataFrame(f1_scores_by_n)

def calculate_hybrid_metrics(predictions_df, n_optimal):
    """
    Calculate performance metrics for the hybrid pipeline using the optimal N.
    """
    print(f"Calculating metrics for optimal N = {n_optimal}")
    
    subjects = predictions_df['subject_id'].unique()
    fold_results = []

    for subject_id in subjects:
        subject_df = predictions_df[predictions_df['subject_id'] == subject_id].sort_values('sample_index')
        
        # Generate hybrid predictions for the subject
        no_pt_preds = subject_df['pred_no_pt'].iloc[:n_optimal]
        pt_preds = subject_df['pred_pt'].iloc[n_optimal:]
        y_hybrid_pred = pd.concat([no_pt_preds, pt_preds])
        y_true = subject_df['true_label']

        # Calculate metrics for the subject
        fold_metrics = {
            'subject_id': subject_id,
            'accuracy': accuracy_score(y_true, y_hybrid_pred),
            'precision': precision_score(y_true, y_hybrid_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_hybrid_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_hybrid_pred, average='weighted', zero_division=0),
            'num_samples': len(y_true)
        }
        fold_results.append(fold_metrics)
        
    results_df = pd.DataFrame(fold_results)
    
    # Calculate overall weighted metrics
    weighted_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        weighted_metrics[f'weighted_{metric}'] = np.average(
            results_df[metric], weights=results_df['num_samples']
        )
    
    print("Metrics calculation complete.")
    return results_df, weighted_metrics

def compare_preprocessing_methods():
    """Compare standard and online preprocessing methods"""
    
    # Load both datasets
    print("=== Loading Standard Preprocessed Dataset ===")
    try:
        X_standard, y_standard = load_data(STANDARD_DATASET)
        # Filter out class 2 samples
        mask_standard = y_standard != 2
        X_standard = X_standard[mask_standard]
        y_standard = y_standard[mask_standard]
    except FileNotFoundError:
        print(f"Standard dataset not found at {STANDARD_DATASET}")
        return

    print("\n=== Loading Online Preprocessed Dataset ===")
    try:
        X_online, y_online = load_data(ONLINE_DATASET)
        # Filter out class 2 samples
        mask_online = y_online != 2
        X_online = X_online[mask_online]
        y_online = y_online[mask_online]
    except FileNotFoundError:
        print(f"Online dataset not found at {ONLINE_DATASET}")
        print("Please run online_preprocessing.py and online_preprocessing_dataset.py first")
        return
    
    # Compare basic dataset properties
    print("\n=== Dataset Comparison ===")
    print(f"Standard dataset: {X_standard['sample'].shape}")
    print(f"Online dataset: {X_online['sample'].shape}")
    print(f"Standard subjects: {sorted(np.unique(X_standard['subject_id']))}")
    print(f"Online subjects: {sorted(np.unique(X_online['subject_id']))}")
    
    # Run evaluations
    print("\n=== Evaluating Standard Preprocessing ===")
    predictions_standard = run_hybrid_evaluation(X_standard, y_standard, "Standard")
    
    print("\n=== Evaluating Online Preprocessing ===")
    predictions_online = run_hybrid_evaluation(X_online, y_online, "Online")
    
    # Find optimal N for both
    print("\n=== Finding Optimal N for Standard ===")
    n_optimal_standard, f1_scores_standard = find_optimal_n(predictions_standard)
    
    print("\n=== Finding Optimal N for Online ===")
    n_optimal_online, f1_scores_online = find_optimal_n(predictions_online)
    
    # Calculate metrics for both
    results_standard, metrics_standard = calculate_hybrid_metrics(predictions_standard, n_optimal_standard)
    results_online, metrics_online = calculate_hybrid_metrics(predictions_online, n_optimal_online)
    
    # Create comparison plots
    print("\n=== Creating Comparison Plots ===")
    
    # Plot F1 scores by N for both methods
    plt.figure(figsize=(12, 6))
    plt.plot(f1_scores_standard['n'], f1_scores_standard['f1_score'], 
             label=f'Standard (Optimal N={n_optimal_standard})', marker='o', alpha=0.7)
    plt.plot(f1_scores_online['n'], f1_scores_online['f1_score'], 
             label=f'Online (Optimal N={n_optimal_online})', marker='s', alpha=0.7)
    plt.axvline(n_optimal_standard, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(n_optimal_online, color='orange', linestyle='--', alpha=0.5)
    plt.title('F1 Score vs. N: Standard vs Online Preprocessing')
    plt.xlabel('N (Number of samples for non-PT pipeline)')
    plt.ylabel('Overall F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(OUTPUT_DIR, 'f1_comparison_preprocessing_methods.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1 comparison plot saved to {plot_path}")
    
    # Plot performance comparison by metric
    metrics_comparison = pd.DataFrame({
        'Standard': [metrics_standard[f'weighted_{m}'] for m in ['accuracy', 'precision', 'recall', 'f1']],
        'Online': [metrics_online[f'weighted_{m}'] for m in ['accuracy', 'precision', 'recall', 'f1']]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1'])
    
    plt.figure(figsize=(10, 6))
    metrics_comparison.plot(kind='bar', width=0.8)
    plt.title('Performance Comparison: Standard vs Online Preprocessing')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45)
    plt.legend(title='Preprocessing Method')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, 'metrics_comparison_preprocessing.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to {comparison_path}")
    
    # Save detailed results
    print("\n=== Saving Results ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary metrics
    summary = {
        'standard_optimal_n': n_optimal_standard,
        'online_optimal_n': n_optimal_online,
        'standard_metrics': metrics_standard,
        'online_metrics': metrics_online,
        'standard_dataset_shape': X_standard['sample'].shape,
        'online_dataset_shape': X_online['sample'].shape,
        'timestamp': timestamp
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(OUTPUT_DIR, f'preprocessing_comparison_summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    
    # Print final comparison
    print("\n=== FINAL COMPARISON ===")
    print(f"Standard Preprocessing:")
    print(f"  Optimal N: {n_optimal_standard}")
    print(f"  Weighted F1: {metrics_standard['weighted_f1']:.4f}")
    print(f"  Weighted Accuracy: {metrics_standard['weighted_accuracy']:.4f}")
    
    print(f"Online Preprocessing:")
    print(f"  Optimal N: {n_optimal_online}")
    print(f"  Weighted F1: {metrics_online['weighted_f1']:.4f}")
    print(f"  Weighted Accuracy: {metrics_online['weighted_accuracy']:.4f}")
    
    # Check if results are similar
    f1_diff = abs(metrics_standard['weighted_f1'] - metrics_online['weighted_f1'])
    accuracy_diff = abs(metrics_standard['weighted_accuracy'] - metrics_online['weighted_accuracy'])
    
    print(f"\nDifferences:")
    print(f"  F1 difference: {f1_diff:.4f}")
    print(f"  Accuracy difference: {accuracy_diff:.4f}")
    
    if f1_diff < 0.02 and accuracy_diff < 0.02:
        print("✅ Online preprocessing results are very similar to standard preprocessing!")
    elif f1_diff < 0.05 and accuracy_diff < 0.05:
        print("⚠️  Online preprocessing results are reasonably similar to standard preprocessing.")
    else:
        print("❌ Online preprocessing results differ significantly from standard preprocessing.")

def main():
    """Main function"""
    compare_preprocessing_methods()

if __name__ == '__main__':
    main()