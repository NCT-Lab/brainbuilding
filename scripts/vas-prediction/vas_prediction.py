import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pyriemann.estimation import Covariances
import warnings
from tqdm import tqdm

# Configure Matplotlib for editable text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy')
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = 'vas-prediction/results'
POSITIVE_CLASS_LABEL = 1 # Label designated for motor imagery
RANDOM_STATE = 42 # For reproducibility

# --- Helper Functions ---

def load_data(filepath):
    """Loads structured numpy array from .npy file."""
    absolute_filepath = os.path.abspath(filepath)
    if not os.path.exists(absolute_filepath):
        raise FileNotFoundError(f"Data file not found at {absolute_filepath}")

    data = np.load(absolute_filepath, allow_pickle=True)
    required_fields = ['sample', 'label', 'subject_id', 'sample_weight', 'is_background']
    if not all(field in data.dtype.names for field in required_fields):
        raise ValueError(f"Loaded data missing required fields. Found: {data.dtype.names}. Required: {required_fields}")
    if data.size == 0:
        print(f"Warning: Loaded data from {absolute_filepath} is empty.")
    return data

def flatten_covariances(cov_matrices):
    """Flattens the upper triangle (including diagonal) of covariance matrices."""
    n_epochs, n_channels, _ = cov_matrices.shape
    # Get indices for the upper triangle
    inds = np.triu_indices(n_channels)
    n_features = len(inds[0])
    flat_covs = np.empty((n_epochs, n_features))
    for i, cov in enumerate(cov_matrices):
        flat_covs[i] = cov[inds]
    return flat_covs

def plot_regression_metrics(metrics_df, metric_col='r2', output_path=None):
    """Generates and saves a bar plot of a specified regression metric per subject, sorted by the metric."""
    if metrics_df.empty or metric_col not in metrics_df.columns:
        print(f"Metrics DataFrame is empty or missing '{metric_col}', skipping plot.")
        return

    # Sort subjects by the metric in descending order
    sorted_df = metrics_df.sort_values(metric_col, ascending=False).copy()
    sorted_df['subject_id'] = sorted_df['subject_id'].astype(str)
    subject_order = sorted_df['subject_id'].tolist()

    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.barplot(data=sorted_df, x='subject_id', y=metric_col, order=subject_order, palette='viridis')

    metric_name = metric_col.replace('_', ' ').title()
    plt.title(f'Per-Subject VAS Prediction {metric_name} (Sorted)')
    plt.xlabel('Subject ID')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Saved {metric_name} plot to: {output_path}")
    plt.close()


# --- Main Script Logic ---

def main():
    # 1. Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved in: {os.path.abspath(RESULTS_DIR)}")

    # 2. Load data
    print(f"Loading data from: {DATASET_FNAME}")
    data = load_data(DATASET_FNAME)
    if data.size == 0:
         print("Loaded data is empty. Exiting.")
         sys.exit(0)
    print(f"Loaded data with shape: {data.shape} and dtype: {data.dtype}")

    # 3. Filter out background samples
    if 'is_background' in data.dtype.names:
        background_mask = data['is_background']
        original_count = len(data)
        data = data[~background_mask]
        print(f"Filtered {np.sum(background_mask)} background samples. Remaining: {len(data)}")
        if len(data) == 0:
            print("No non-background samples found. Exiting.")
            sys.exit(0)

    # 4. Get unique subjects
    subject_ids = np.unique(data['subject_id'])
    print(f"Found {len(subject_ids)} subjects: {subject_ids.tolist()}")

    # 5. Initialize storage for results
    all_predictions = []
    subject_metrics_list = []

    # 6. Per-subject LOOCV for VAS prediction
    for subject_id in subject_ids:
        print(f"\nProcessing subject {subject_id}...")
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]

        # Filter for motor imagery samples ONLY
        mi_mask = subject_data['label'] == POSITIVE_CLASS_LABEL
        mi_data = subject_data[mi_mask]
        n_mi_samples = len(mi_data)

        if n_mi_samples < 2:
            print(f"Skipping subject {subject_id}: Insufficient motor imagery samples ({n_mi_samples}). Need at least 2 for LOOCV.")
            continue
        print(f"Found {n_mi_samples} motor imagery samples for subject {subject_id}.")

        # Extract EEG data and true VAS scores for motor imagery samples
        X_mi = mi_data['sample']
        y_true_vas = mi_data['sample_weight']

        # Compute covariance matrices
        cov_estimator = Covariances(estimator='oas')
        covs_mi = cov_estimator.fit_transform(X_mi)

        # Flatten covariances for feature vectors
        X_flat = flatten_covariances(covs_mi)

        # Define the Regressor
        regressor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

        loo = LeaveOneOut()
        subject_loocv_preds = []
        y_pred_vas_list = np.zeros(n_mi_samples) # Array to store predictions

        print(f"Running LOOCV for {n_mi_samples} samples...")
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(X_flat), total=n_mi_samples, desc=f"Subject {subject_id} LOOCV", unit="sample")):
            X_train, X_test = X_flat[train_index], X_flat[test_index]
            y_train = y_true_vas[train_index]

            # Fit the regressor
            regressor.fit(X_train, y_train)

            # Predict VAS for the left-out sample
            y_pred = regressor.predict(X_test)[0]
            y_pred_vas_list[i] = y_pred # Store prediction in order

            # Store individual prediction details
            subject_loocv_preds.append({
                'subject_id': subject_id,
                'true_vas': y_true_vas[test_index][0],
                'predicted_vas': y_pred
            })

        all_predictions.extend(subject_loocv_preds)

        # Calculate overall metrics for the subject using all LOOCV predictions
        mae = mean_absolute_error(y_true_vas, y_pred_vas_list)
        r2 = r2_score(y_true_vas, y_pred_vas_list)

        print(f"Finished subject {subject_id}. MAE: {mae:.4f}, R2: {r2:.4f}")
        subject_metrics_list.append({
            'subject_id': subject_id,
            'mae': mae,
            'r2': r2,
            'n_mi_samples': n_mi_samples
        })

    # 7. Consolidate and Save Results
    if not all_predictions:
        print("\nNo predictions were made (possibly no subjects had enough MI samples). Exiting.")
        sys.exit(0)

    overall_preds_df = pd.DataFrame(all_predictions)
    subject_metrics_df = pd.DataFrame(subject_metrics_list)

    # Save predictions
    overall_preds_path = os.path.join(RESULTS_DIR, 'overall_vas_predictions.csv')
    overall_preds_df.to_csv(overall_preds_path, index=False)
    print(f"\nSaved overall VAS predictions ({len(overall_preds_df)} rows) to: {overall_preds_path}")

    # Save metrics
    subject_metrics_path = os.path.join(RESULTS_DIR, 'subject_vas_metrics.csv')
    subject_metrics_df.to_csv(subject_metrics_path, index=False)
    print(f"Saved subject VAS metrics ({len(subject_metrics_df)} rows) to: {subject_metrics_path}")

    # 8. Generate and Save Plot
    r2_plot_path = os.path.join(RESULTS_DIR, 'subject_r2_scores_plot.pdf')
    plot_regression_metrics(subject_metrics_df, metric_col='r2', output_path=r2_plot_path)


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main() 