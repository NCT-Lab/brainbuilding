import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from tqdm import tqdm

# Configure Matplotlib for editable text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join('..' ,'data', 'preprocessed', 'motor-imagery-2.npy')
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = 'results' # Save results in ./results relative to script
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0
EXPECTED_SAMPLES_PER_SUBJECT = 120
EXPECTED_PAIRS_PER_SUBJECT = EXPECTED_SAMPLES_PER_SUBJECT // 2
CSP_N_COMPONENTS = 10
RANDOM_STATE = 42

# --- Helper Functions ---

def load_data(filepath):
    """Loads structured numpy array from .npy file."""
    absolute_filepath = os.path.abspath(filepath)
    if not os.path.exists(absolute_filepath):
        script_dir = os.path.dirname(__file__)
        alt_path = os.path.abspath(os.path.join(script_dir, '..', DEFAULT_DATASET_PATH))
        if os.path.exists(alt_path):
            print(f"Data file not found at {absolute_filepath}, using alternative default path: {alt_path}")
            absolute_filepath = alt_path
        else:
            raise FileNotFoundError(f"Data file not found at {absolute_filepath} or {alt_path}")

    data = np.load(absolute_filepath, allow_pickle=True)
    required_fields = ['sample', 'label', 'subject_id', 'sample_weight', 'is_background']
    if not all(field in data.dtype.names for field in required_fields):
        raise ValueError(f"Loaded data missing required fields. Found: {data.dtype.names}. Required: {required_fields}")
    if data.size == 0:
        print(f"Warning: Loaded data from {absolute_filepath} is empty.")
    return data

def plot_regression_metrics(metrics_df, metric_col='r2', output_path=None):
    """Generates and saves a bar plot of a specified regression metric per subject, sorted by the metric."""
    if metrics_df.empty or metric_col not in metrics_df.columns:
        print(f"Metrics DataFrame is empty or missing '{metric_col}', skipping plot.")
        return

    sorted_df = metrics_df.sort_values(metric_col, ascending=False).copy()
    sorted_df['subject_id'] = sorted_df['subject_id'].astype(str)
    subject_order = sorted_df['subject_id'].tolist()

    plt.figure(figsize=(12, 7))
    sns.barplot(data=sorted_df, x='subject_id', y=metric_col, order=subject_order, palette='viridis')

    metric_name = metric_col.replace('_', ' ').title()
    plt.title(f'Per-Subject VAS Prediction {metric_name} using CSP Difference (Sorted)') # Updated title
    plt.xlabel('Subject ID')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Saved {metric_name} plot to: {output_path}")
    plt.close()

# --- Main Script Logic ---
def main():
    # 1. Create results directory
    script_dir = os.path.dirname(__file__)
    results_full_path = os.path.join(script_dir, RESULTS_DIR)
    os.makedirs(results_full_path, exist_ok=True)
    print(f"Results will be saved in: {results_full_path}")

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
        data = data[~background_mask]
        print(f"Filtered {np.sum(background_mask)} background samples. Remaining: {len(data)}")
        if len(data) == 0:
            print("No non-background samples found. Exiting.")
            sys.exit(0)

    # 4. Get unique subjects
    subject_ids = np.unique(data['subject_id'])
    print(f"Found {len(subject_ids)} subjects: {subject_ids.tolist()}")

    # 5. Initialize storage for results
    all_loocv_predictions = []
    subject_metrics_list = []
    cov_estimator = Covariances(estimator='oas')

    # 6. Process each subject
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]
        n_samples = len(subject_data)

        if n_samples != EXPECTED_SAMPLES_PER_SUBJECT:
            print(f"Warning: Subject {subject_id} has {n_samples} samples, expected {EXPECTED_SAMPLES_PER_SUBJECT}. Skipping subject.")
            continue

        # Extract EEG data, labels, and VAS scores
        X_subj = subject_data['sample']
        y_subj = subject_data['label']
        vas_scores_all = subject_data['sample_weight']

        # --- CSP Calculation ---
        try:
            covs_subj = cov_estimator.fit_transform(X_subj)
            csp = CSP(nfilter=CSP_N_COMPONENTS, log=True, metric='riemann')
            csp.fit(covs_subj, y_subj)
            csp_features_subj = csp.transform(covs_subj)
        except Exception as e:
            print(f"Error processing CSP for subject {subject_id}: {e}. Skipping subject.")
            continue
        # -----------------------

        X_pairs_diff = []
        y_pairs_vas = []

        # Create pairs and calculate difference vectors
        for i in range(0, n_samples, 2):
            csp_feat_rest = csp_features_subj[i]
            csp_feat_mi = csp_features_subj[i+1]
            vas_mi = vas_scores_all[i+1]

            # Calculate difference vector (MI - Rest)
            diff_vector = csp_feat_mi - csp_feat_rest

            X_pairs_diff.append(diff_vector)
            y_pairs_vas.append(vas_mi)

        X_pairs_diff = np.array(X_pairs_diff)
        y_pairs_vas = np.array(y_pairs_vas)
        n_pairs = len(X_pairs_diff)

        if n_pairs < 2:
             print(f"Warning: Subject {subject_id} resulted in < 2 pairs ({n_pairs}). Skipping LOOCV.")
             continue

        # --- LOOCV on Pairs ---
        regressor = RandomForestRegressor(n_estimators=20, random_state=RANDOM_STATE, n_jobs=-1)
        loo = LeaveOneOut()
        subject_preds_temp = []
        y_pred_vas_loocv = np.zeros(n_pairs)

        print(f"Running LOOCV for {n_pairs} pairs...")
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(X_pairs_diff), total=n_pairs, desc=f"Subject {subject_id} Pair LOOCV", unit="pair", leave=False)):
            X_train, X_test = X_pairs_diff[train_index], X_pairs_diff[test_index]
            y_train = y_pairs_vas[train_index]
            true_vas_test = y_pairs_vas[test_index][0]

            try:
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)[0]
                y_pred_vas_loocv[i] = y_pred

                subject_preds_temp.append({
                    'subject_id': subject_id,
                    'pair_index': test_index[0], # Index of the pair being tested
                    'true_vas': true_vas_test,
                    'predicted_vas': y_pred
                })
            except Exception as e:
                 print(f"Error during LOOCV fold {i} for subject {subject_id}: {e}")
                 # Optionally store NaN or skip this prediction
                 y_pred_vas_loocv[i] = np.nan # Mark as invalid if error occurs
                 subject_preds_temp.append({
                    'subject_id': subject_id,
                    'pair_index': test_index[0],
                    'true_vas': true_vas_test,
                    'predicted_vas': np.nan
                })

        all_loocv_predictions.extend(subject_preds_temp)

        # Calculate overall metrics for the subject using valid LOOCV predictions
        valid_mask = ~np.isnan(y_pred_vas_loocv)
        if np.sum(valid_mask) > 0:
            mae = mean_absolute_error(y_pairs_vas[valid_mask], y_pred_vas_loocv[valid_mask])
            r2 = r2_score(y_pairs_vas[valid_mask], y_pred_vas_loocv[valid_mask])
            print(f"Finished subject {subject_id}. MAE: {mae:.4f}, R2: {r2:.4f} ({np.sum(valid_mask)}/{n_pairs} valid folds)")
        else:
            mae = np.nan
            r2 = np.nan
            print(f"Finished subject {subject_id}. No valid LOOCV folds completed.")

        subject_metrics_list.append({
            'subject_id': subject_id,
            'mae': mae,
            'r2': r2,
            'n_pairs': n_pairs,
            'n_valid_folds': int(np.sum(valid_mask))
        })
        # ---------------------

    # 7. Consolidate and Save Results
    if not all_loocv_predictions:
        print("\nNo LOOCV predictions were made. Exiting.")
        sys.exit(0)

    overall_preds_df = pd.DataFrame(all_loocv_predictions)
    subject_metrics_df = pd.DataFrame(subject_metrics_list)

    # Save predictions
    overall_preds_path = os.path.join(results_full_path, 'overall_loocv_vas_predictions.csv')
    overall_preds_df.to_csv(overall_preds_path, index=False)
    print(f"\nSaved overall LOOCV VAS predictions ({len(overall_preds_df)} rows) to: {overall_preds_path}")

    # Save metrics
    subject_metrics_path = os.path.join(results_full_path, 'subject_vas_regression_metrics.csv')
    subject_metrics_df.to_csv(subject_metrics_path, index=False)
    print(f"Saved subject VAS regression metrics ({len(subject_metrics_df)} rows) to: {subject_metrics_path}")

    # 8. Generate and Save Plot
    r2_plot_path = os.path.join(results_full_path, 'subject_r2_scores_plot.pdf')
    plot_regression_metrics(subject_metrics_df, metric_col='r2', output_path=r2_plot_path)


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main() 