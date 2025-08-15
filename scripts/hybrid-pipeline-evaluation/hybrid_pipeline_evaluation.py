import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from datetime import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.brainbuilding.pipelines import (
    WHITENING, BACKGROUND_FILTER, AUGMENTED_OAS_COV,
    PT_CSP_LOG, SIMPLE_CSP, SVC_CLASSIFICATION
)
from src.brainbuilding.evaluation import evaluate_pipeline

# --- Configuration ---
INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", 'data/preprocessed')
INPUT_DATASET_FNAME = os.getenv("INPUT_DATASET_NAME", 'motor-imagery-2.npy')
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

    print("Data loaded successfully.")
    return X, y


def create_pt_pipeline_steps():
    """Creates the pipeline steps with Parallel Transport."""
    return WHITENING + BACKGROUND_FILTER + AUGMENTED_OAS_COV + PT_CSP_LOG + SVC_CLASSIFICATION


def create_no_pt_pipeline_steps():
    """Creates the pipeline steps without Parallel Transport."""
    return WHITENING + BACKGROUND_FILTER + AUGMENTED_OAS_COV + SIMPLE_CSP + SVC_CLASSIFICATION


def run_hybrid_evaluation(X, y):
    """
    Evaluate both pipelines and collect their predictions.
    """
    print("Running hybrid evaluation...")
    
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

    for n in tqdm(range(max_samples + 1), desc="Finding Optimal N"):
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
    
    # Plot F1 scores by N
    f1_df = pd.DataFrame(f1_scores_by_n)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=f1_df, x='n', y='f1_score')
    plt.title('F1 Score vs. N (Switching Point)')
    plt.xlabel('N (Number of samples for non-PT pipeline)')
    plt.ylabel('Overall F1 Score')
    plt.axvline(best_n, color='r', linestyle='--', label=f'Optimal N = {best_n}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, 'f1_vs_n_plot.png')
    plt.savefig(plot_path)
    print(f"F1 vs. N plot saved to {plot_path}")
    plt.close()
    
    return best_n


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


def plot_results(results_df, optimal_n, output_path):
    """Plot and save the evaluation results."""
    print(f"Plotting results and saving to {output_path}")
    
    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")
    
    metric_cols = ['accuracy', 'precision', 'recall', 'f1']
    results_df = results_df.sort_values('f1', ascending=False)
    
    x = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metric_cols):
        offset = width * (i - len(metric_cols) / 2 + 0.5)
        plt.bar(x + offset, results_df[metric], width, label=metric.capitalize())

    plt.xlabel('Subject ID')
    plt.ylabel('Score')
    plt.title(f'Hybrid Pipeline Performance by Subject (Optimal N={optimal_n})')
    plt.xticks(x, results_df['subject_id'], rotation=90)
    plt.legend()
    plt.ylim(0, 1)
    
    # Add mean lines
    for i, metric in enumerate(metric_cols):
        mean_val = results_df[metric].mean()
        plt.axhline(y=mean_val, color=f'C{i}', linestyle='--', alpha=0.7, label=f'Mean {metric.capitalize()}: {mean_val:.3f}')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Create individual plots for each metric
    for metric in metric_cols:
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='subject_id', y=metric, data=results_df.sort_values(metric, ascending=False), palette='viridis')
        ax.set_title(f'Hybrid Pipeline {metric.capitalize()} by Subject (N={optimal_n})')
        ax.set_xlabel('Subject ID')
        ax.set_ylabel(metric.capitalize())
        
        mean_val = results_df[metric].mean()
        plt.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
        plt.legend()
        
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        
        plt.tight_layout()
        metric_plot_path = os.path.join(os.path.dirname(output_path), f'hybrid_{metric}.png')
        plt.savefig(metric_plot_path)
        plt.close()

def save_results(results, output_path):
    """Save results to a pickle file and CSV."""
    # Convert DataFrame to dict for serialization
    results['fold_results'] = results['fold_results'].to_dict(orient='records')
    results['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full results saved to: {output_path}")
    
    csv_path = output_path.replace('.pkl', '.csv')
    pd.DataFrame(results['fold_results']).to_csv(csv_path, index=False)
    print(f"Per-subject metrics saved to CSV: {csv_path}")

def main():
    """Main function to run the evaluation."""
    input_path = os.path.join(INPUT_DATA_DIR, INPUT_DATASET_FNAME)
    
    X_structured, y = load_data(input_path)
    
    # Filter out class 2 samples as is common in other scripts
    mask = y != 2
    X = X_structured[mask]
    y = y[mask]
    
    predictions_df = run_hybrid_evaluation(X, y)
    
    n_optimal = find_optimal_n(predictions_df)
    
    results_df, overall_metrics = calculate_hybrid_metrics(predictions_df, n_optimal)

    # --- Save and plot results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plotting
    plot_path = os.path.join(OUTPUT_DIR, f'hybrid_pipeline_results_{timestamp}.png')
    plot_results(results_df, n_optimal, plot_path)
    
    # Saving
    results_to_save = {
        'fold_results': results_df,
        'overall_metrics': overall_metrics,
        'optimal_n': n_optimal
    }
    results_path = os.path.join(OUTPUT_DIR, f'hybrid_pipeline_results_{timestamp}.pkl')
    save_results(results_to_save, results_path)


if __name__ == '__main__':
    main() 