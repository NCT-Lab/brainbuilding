import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from scipy.stats import pearsonr
import warnings
from tqdm import tqdm

# --- Constants ---
# Use environment variable or default relative path within the project structure
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy')
# Allow overriding via env var, e.g., export DATASET_FNAME='path/to/your/data.npy'
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = 'vas-correlation/results'
POSITIVE_CLASS_LABEL = 1 # Label designated for motor imagery
CSP_N_COMPONENTS = 10 # Number of CSP components


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

def plot_metrics(metrics_df, output_path):
    """Generates and saves a bar plot of metrics per subject, sorted by F1 score."""
    if metrics_df.empty:
        print("Metrics DataFrame is empty, skipping plot generation.")
        return

    # Sort subjects by F1 score in descending order for plotting
    sorted_df = metrics_df.sort_values('f1', ascending=False).copy()
    # Ensure subject_id is treated as a categorical variable for ordering
    sorted_df['subject_id'] = sorted_df['subject_id'].astype(str)
    subject_order = sorted_df['subject_id'].tolist()

    # Melt the DataFrame for Seaborn compatibility
    metrics_to_plot = ['precision', 'recall', 'accuracy', 'f1', 'auc', 'vas_correlation']
    # Filter out any metrics that might be missing (e.g., if AUC/correlation failed for all)
    cols_to_melt = [col for col in metrics_to_plot if col in sorted_df.columns]
    if not cols_to_melt:
        print("No valid metric columns found to plot.")
        return

    melted_df = sorted_df.melt(id_vars='subject_id',
                               value_vars=cols_to_melt,
                               var_name='metric',
                               value_name='value')

    # Create the plot
    plt.figure(figsize=(15, 8))
    plot = sns.barplot(data=melted_df, x='subject_id', y='value', hue='metric', order=subject_order)

    plt.title('Per-Subject Classification Metrics and VAS Correlation (Sorted by F1)')
    plt.xlabel('Subject ID')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45, ha='right')
    # Place legend outside the plot
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Adjust layout to prevent clipping
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust right boundary for legend

    # Save the plot directly, errors will propagate
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close() # Close the figure to free memory


# --- Main Script Logic ---

def main():
    # 1. Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved in: {os.path.abspath(RESULTS_DIR)}")

    # 2. Load data
    print(f"Loading data from: {DATASET_FNAME}")
    data = load_data(DATASET_FNAME)
    if data.size == 0:
         print("Loaded data is empty. Exiting.")
         sys.exit(0) # Exit gracefully if data is empty
    print(f"Loaded data with shape: {data.shape} and dtype: {data.dtype}")

    # 3. Filter out background samples
    if 'is_background' in data.dtype.names:
        background_mask = data['is_background']
        original_count = len(data)
        data = data[~background_mask]
        filtered_count = np.sum(background_mask)
        print(f"Filtered {filtered_count} background samples. Remaining samples: {len(data)} (out of {original_count})")
        if len(data) == 0:
            print("No non-background samples found after filtering. Exiting.")
            sys.exit(0)
    else:
        print("Warning: 'is_background' field not found in data. Proceeding without background filtering.")

    # 4. Get unique subjects
    subject_ids = np.unique(data['subject_id'])
    print(f"Found {len(subject_ids)} subjects: {subject_ids.tolist()}")

    # 5. Initialize storage for results from all subjects
    all_predictions = []

    # 6. Per-subject Leave-One-Out Cross-Validation (LOOCV)
    for subject_id in subject_ids:
        print(f"Processing subject {subject_id}...")
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]

        # Extract relevant fields for this subject
        X_subj = subject_data['sample'] # Expected shape: (n_epochs, n_channels, n_times)
        y_subj = subject_data['label']
        weights_subj = subject_data['sample_weight'] # VAS scores

        n_samples = len(X_subj)
        unique_classes = np.unique(y_subj)

        # Basic checks before starting LOOCV for the subject
        if n_samples < 2:
            print(f"Skipping subject {subject_id}: Insufficient samples ({n_samples}). Need at least 2 for LOOCV.")
            continue
        if len(unique_classes) < 2:
             print(f"Skipping subject {subject_id}: Only one class ({unique_classes}) present. Cannot train classifier.")
             continue

        # Define the MNE/Sklearn pipeline
        # Use robust estimators ('oas' for covariance, 'ledoit_wolf' regularization for CSP)
        cov_estimator = Covariances(estimator='oas')
        csp = CSP(nfilter=CSP_N_COMPONENTS, log=True, metric='riemann')
        svm = SVC(kernel='rbf', gamma='scale', probability=True, C=0.696) # Added balanced class weight

        clf = make_pipeline(cov_estimator, csp, svm)

        loo = LeaveOneOut()
        subject_predictions = []
        processed_folds = 0

        # Iterate through each sample using LOOCV indices with tqdm
        print(f"Running LOOCV for {n_samples} samples...") # Info before tqdm bar
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(X_subj), total=n_samples, desc=f"Subject {subject_id} LOOCV", unit="fold")):
            X_train, X_test = X_subj[train_index], X_subj[test_index]
            y_train, y_test = y_subj[train_index], y_subj[test_index]
            current_weight = weights_subj[test_index][0] # VAS score for the test sample

            pred_label = -99 # Placeholder for skipped folds
            pred_proba = np.nan

            # Ensure the training split has multiple classes
            if len(np.unique(y_train)) < 2:
                # This is a data limitation, not an error, so keep the skip logic
                # No print here, tqdm handles progress
                pass # Skip this fold
            else:
                # Fit the pipeline on the training data - Errors will propagate
                clf.fit(X_train, y_train)

                # Predict class label for the left-out sample
                pred_label = clf.predict(X_test)[0]

                # Predict class probabilities
                proba_estimates = clf.predict_proba(X_test)[0]

                # Get the probability of the positive class
                try:
                    # Find the index corresponding to the positive class in the fitted classifier
                    positive_class_idx = clf.classes_.tolist().index(POSITIVE_CLASS_LABEL)
                    pred_proba = proba_estimates[positive_class_idx]
                except ValueError:
                    # This happens if the positive class was not present in y_train for this fold
                    # Still using try-except here as it's expected data variation, not a code error.
                    # Setting proba to NaN is the desired outcome.
                    print(f"\nWarning: Positive class {POSITIVE_CLASS_LABEL} not in training data for subject {subject_id}, fold {i+1}. Proba set to NaN.")
                    pred_proba = np.nan

                processed_folds += 1 # Increment only if fitted and predicted

            # Store results for this fold (even if skipped)
            subject_predictions.append({
                'subject_id': subject_id,
                'true_label': y_test[0],
                'predicted_label': pred_label, # Will be -99 for skipped folds
                'predicted_proba': pred_proba,
                'sample_weight': current_weight
            })

        # Updated summary message
        skipped_folds = n_samples - processed_folds
        print(f"Finished subject {subject_id}. Processed {processed_folds} folds, skipped {skipped_folds} folds due to single class in train split.")
        all_predictions.extend(subject_predictions)

    # 7. Consolidate results into a DataFrame
    overall_results_df = pd.DataFrame(all_predictions)

    # Separate valid predictions from folds that were skipped or had errors
    valid_results_mask = overall_results_df['predicted_label'] != -99
    valid_results_df = overall_results_df[valid_results_mask].copy()
    # Convert predicted label to int for valid results
    if not valid_results_df.empty:
        valid_results_df['predicted_label'] = valid_results_df['predicted_label'].astype(int)

    if valid_results_df.empty:
        print("No valid predictions were obtained across all subjects. Cannot calculate metrics or plot.")
        # Save the raw (error/skip) results for debugging
        raw_results_path = os.path.join(RESULTS_DIR, 'overall_predictions_raw.csv')
        overall_results_df.to_csv(raw_results_path, index=False)
        print(f"Saved raw prediction attempts (including skips/errors) to: {raw_results_path}")
        sys.exit(0) # Exit gracefully

    print(f"Total valid predictions collected across all subjects: {len(valid_results_df)}")

    # 8. Calculate per-subject metrics using only the valid results
    subject_metrics = []
    grouped = valid_results_df.groupby('subject_id')

    for subject_id, group in grouped:
        y_true = group['true_label']
        y_pred = group['predicted_label']

        # Prepare data for AUC (requires probabilities and handles potential NaNs)
        proba_data = group[['true_label', 'predicted_proba']].dropna()
        y_true_for_auc = proba_data['true_label']
        y_proba_for_auc = proba_data['predicted_proba']

        # Calculate standard metrics
        accuracy = accuracy_score(y_true, y_pred)
        # Handle cases with no positive predictions/labels gracefully
        precision = precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)

        # Calculate AUC if possible - Errors will propagate
        auc = np.nan
        if len(np.unique(y_true_for_auc)) >= 2:
            auc = roc_auc_score(y_true_for_auc, y_proba_for_auc)
        else:
            if len(y_proba_for_auc) > 0: # Only warn if there were probabilities to use
                 print(f"Skipping AUC for subject {subject_id}: Only one class present in data with valid probabilities.")


        # Calculate VAS Correlation for positive true labels with valid probabilities - Errors will propagate
        correlation = np.nan
        p_value = np.nan
        # Filter for positive class samples where prediction probability is not NaN
        positive_samples = group[(group['true_label'] == POSITIVE_CLASS_LABEL) & group['predicted_proba'].notna()].copy()
        corr_data = positive_samples[['predicted_proba', 'sample_weight']].dropna() # Ensure weight is also not NaN

        if len(corr_data) >= 2:
            # Check for constant series which also prevent correlation calculation
            if corr_data['predicted_proba'].nunique() > 1 and corr_data['sample_weight'].nunique() > 1:
                 correlation, p_value = pearsonr(corr_data['predicted_proba'], corr_data['sample_weight'])
            else:
                 print(f"Could not calculate VAS correlation for subject {subject_id}: Predicted probabilities or VAS scores are constant for positive samples.")
        else:
            print(f"Not enough valid positive samples (found {len(corr_data)}, need >= 2) for VAS correlation for subject {subject_id}.")


        subject_metrics.append({
            'subject_id': subject_id,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'vas_correlation': correlation,
            # Add counts for context
            'n_samples_total': len(subject_data), # Total samples for subject before LOOCV
            'n_samples_valid_pred': len(group), # Samples with successful predictions
            'n_positive_samples': sum(group['true_label'] == POSITIVE_CLASS_LABEL)
        })

    subject_metrics_df = pd.DataFrame(subject_metrics)

    # 9. Save results to CSV files
    overall_results_path = os.path.join(RESULTS_DIR, 'overall_valid_predictions.csv')
    subject_metrics_path = os.path.join(RESULTS_DIR, 'subject_metrics.csv')
    raw_results_path = os.path.join(RESULTS_DIR, 'overall_predictions_raw.csv') # Save all attempts

    valid_results_df.to_csv(overall_results_path, index=False)
    print(f"Saved overall VALID predictions ({len(valid_results_df)} rows) to: {overall_results_path}")
    overall_results_df.to_csv(raw_results_path, index=False)
    print(f"Saved raw prediction attempts including skips/errors ({len(overall_results_df)} rows) to: {raw_results_path}")

    if not subject_metrics_df.empty:
         subject_metrics_df.to_csv(subject_metrics_path, index=False)
         print(f"Saved subject metrics ({len(subject_metrics_df)} rows) to: {subject_metrics_path}")

         # 10. Generate and save the metrics plot
         print("Generating metrics plot...")
         plot_path = os.path.join(RESULTS_DIR, 'subject_metrics_plot.pdf')
         plot_metrics(subject_metrics_df, plot_path)
         print(f"Saved metrics plot to: {plot_path}")
    else:
         print("No subject metrics were calculated, skipping saving metrics file and plot.")


if __name__ == "__main__":
    # Configure warnings for cleaner output
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # Optional: Filter specific runtime warnings from sklearn/scipy if they become excessive
    # warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
    # warnings.filterwarnings("ignore", message="Precision is ill-defined") # From sklearn.metrics
    # warnings.filterwarnings("ignore", message="Recall is ill-defined") # From sklearn.metrics

    main()
