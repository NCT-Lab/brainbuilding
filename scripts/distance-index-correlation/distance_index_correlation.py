import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from scipy.stats import spearmanr, permutation_test
import warnings
from tqdm import tqdm

# Configure Matplotlib for editable text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy') # Path relative to project root
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = os.path.join('scripts', 'distance-index-correlation', 'results') # Path relative to project root
# Path to the F1 results from the VAS correlation script, relative to project root
F1_RESULTS_PATH = os.path.join('scripts', 'vas-correlation', 'results', 'subject_metrics.csv')
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0
EXPECTED_SAMPLES_PER_SUBJECT = 120
EXPECTED_PAIRS_PER_SUBJECT = EXPECTED_SAMPLES_PER_SUBJECT // 2
RANDOM_STATE = 42

# --- Helper Functions ---

def load_data(filepath):
    """Loads structured numpy array from .npy file."""
    absolute_filepath = os.path.abspath(filepath) # Get absolute path based on where script is run
    if not os.path.exists(absolute_filepath):
        # If not found, assume filepath was relative to root and raise error
        raise FileNotFoundError(f"Data file not found at {absolute_filepath} (expected relative to project root)")

    data = np.load(absolute_filepath, allow_pickle=True)
    required_fields = ['sample', 'label', 'subject_id', 'sample_weight', 'is_background']
    if not all(field in data.dtype.names for field in required_fields):
        raise ValueError(f"Loaded data missing required fields. Found: {data.dtype.names}. Required: {required_fields}")
    if data.size == 0:
        print(f"Warning: Loaded data from {absolute_filepath} is empty.")
    return data

def plot_correlation_barplot(metrics_df, output_path):
    """Generates and saves a bar plot of Spearman correlation (Distance vs Pair Index) per subject, sorted,
       with significance asterisks based on permutation p-values."""
    if metrics_df.empty or 'correlation' not in metrics_df.columns or 'p_value_perm' not in metrics_df.columns:
        print("Metrics DataFrame is empty or missing 'correlation'/'p_value_perm', skipping bar plot.")
        return

    sorted_df = metrics_df.sort_values('correlation', ascending=False).copy()
    sorted_df['subject_id'] = sorted_df['subject_id'].astype(str)
    subject_order = sorted_df['subject_id'].tolist()

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=sorted_df, x='subject_id', y='correlation', order=subject_order, palette='viridis')

    plt.title(f'Per-Subject Spearman Correlation (Distance vs. Pair Index) with Permutation Test Significance') # Updated title
    plt.xlabel('Subject ID')
    plt.ylabel('Spearman Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')

    # Add significance asterisks
    heights = [p.get_height() for p in ax.patches]
    positions = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    p_values_map = sorted_df.set_index('subject_id')['p_value_perm'].to_dict()
    y_min, y_max = plt.ylim()
    offset = abs(y_max - y_min) * 0.02

    for i, subject_id_str in enumerate(subject_order):
        p_value = p_values_map.get(subject_id_str)
        if p_value is not None and not np.isnan(p_value):
            if p_value < 0.001:
                sig_symbol = '***'
            elif p_value < 0.01:
                sig_symbol = '**'
            elif p_value < 0.05:
                sig_symbol = '*'
            else:
                sig_symbol = ''
            if sig_symbol:
                bar_height = heights[i]
                text_y = bar_height + offset if bar_height >= 0 else bar_height - offset * 2
                va = 'bottom' if bar_height >= 0 else 'top'
                ax.text(positions[i], text_y, sig_symbol, ha='center', va=va, color='black', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved correlation bar plot with significance to: {output_path}")
    plt.close()

def plot_all_subjects_regressions(pair_data_df, output_path, n_cols=4):
    """Generates a multi-subplot figure showing distance vs pair index regression for each subject."""
    if pair_data_df.empty or not all(c in pair_data_df.columns for c in ['subject_id', 'distance', 'pair_index']):
        print("Pair data DataFrame is empty or missing required columns, skipping multi-subject regression plot.")
        return

    subject_ids = sorted(pair_data_df['subject_id'].unique())
    n_subjects = len(subject_ids)
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, subject_id in enumerate(tqdm(subject_ids, desc="Generating subject regression plots")):
        ax = axes[i]
        subject_data = pair_data_df[pair_data_df['subject_id'] == subject_id].copy()
        subject_data['pair_index'] = pd.to_numeric(subject_data['pair_index'])

        if len(subject_data) > 1:
            # Plot Distance vs Pair Index
            # 1. Plot the LOESS regression line
            sns.regplot(data=subject_data, x='distance', y='pair_index', ax=ax, lowess=True,
                        scatter=False,
                        line_kws={'color': 'black', 'lw': 1.5})
            # 2. Plot the scatter points (no hue needed here)
            sns.scatterplot(data=subject_data, x='distance', y='pair_index', ax=ax,
                            s=25, alpha=0.7)

            ax.set_title(f'Subject {subject_id}')
            ax.set_xlabel('Riemannian Distance')
            ax.set_ylabel('Pair Index') # Updated label
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id}')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Distance between Consecutive Covariance Matrices vs. Pair Index', fontsize=16, y=1.02) # Updated title
    fig.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved multi-subject regression plot to: {output_path}")
    plt.close(fig)

def plot_correlation_f1_barplot(metrics_df, output_path):
    """Generates and saves a bar plot comparing Spearman correlation (Distance vs Pair Index)
       and F1 score per subject, sorted by correlation.
       Adds significance asterisks to correlation bars based on permutation p-values."""
    required_cols = ['subject_id', 'correlation', 'p_value_perm', 'f1']
    if metrics_df.empty or not all(c in metrics_df.columns for c in required_cols):
        print(f"Metrics DataFrame is empty or missing required columns {required_cols}, skipping combined bar plot.")
        return

    # Sort by correlation for subject order
    sorted_df = metrics_df.sort_values('correlation', ascending=False).copy()
    sorted_df['subject_id'] = sorted_df['subject_id'].astype(str)
    subject_order = sorted_df['subject_id'].tolist()

    # Melt for plotting with hue
    melted_df = sorted_df.melt(id_vars=['subject_id', 'p_value_perm'], # Keep p_value for annotations
                               value_vars=['correlation', 'f1'],
                               var_name='metric',
                               value_name='value')

    plt.figure(figsize=(16, 8)) # Wider figure
    ax = sns.barplot(data=melted_df, x='subject_id', y='value', hue='metric',
                     order=subject_order, palette='viridis')

    plt.title(f'Per-Subject Correlation (Dist vs Idx) and Classification F1 Score')
    plt.xlabel('Subject ID')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left') # Move legend out

    # Add significance asterisks ONLY to correlation bars
    # We need to iterate through the patches and check the associated metric
    p_values_map = sorted_df.set_index('subject_id')['p_value_perm'].to_dict()
    y_min, y_max = plt.ylim()
    offset = abs(y_max - y_min) * 0.02

    # Get handles for legend to map hue to metric name
    handles, labels = ax.get_legend_handles_labels()
    hue_order = labels # order corresponds to patch order within a subject group

    num_subjects = len(subject_order)
    num_hues = len(hue_order)

    for i, subject_id_str in enumerate(subject_order):
        p_value = p_values_map.get(subject_id_str)
        if p_value is not None and not np.isnan(p_value):
            # Determine the index of the 'correlation' bar patch for this subject
            try:
                corr_hue_index = hue_order.index('correlation')
                patch_index = i * num_hues + corr_hue_index
                if patch_index < len(ax.patches):
                    target_patch = ax.patches[patch_index]

                    if p_value < 0.001:
                        sig_symbol = '***'
                    elif p_value < 0.01:
                        sig_symbol = '**'
                    elif p_value < 0.05:
                        sig_symbol = '*'
                    else:
                        sig_symbol = ''

                    if sig_symbol:
                        bar_height = target_patch.get_height()
                        bar_x = target_patch.get_x() + target_patch.get_width() / 2
                        text_y = bar_height + offset if bar_height >= 0 else bar_height - offset * 2
                        va = 'bottom' if bar_height >= 0 else 'top'
                        ax.text(bar_x, text_y, sig_symbol, ha='center', va=va, color='black', fontsize=9, fontweight='bold')
            except ValueError:
                 print("Could not find 'correlation' in legend labels to apply significance.")
            except IndexError:
                 print(f"Patch index {patch_index} out of range for subject {subject_id_str}.")

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust right margin for legend
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved combined correlation/F1 bar plot with significance to: {output_path}")
    plt.close()

# --- Spearman Statistic Helper ---
def _spearman_statistic(x, y):
    """Helper function to calculate Spearman correlation for permutation test."""
    res = spearmanr(x, y)
    return res.correlation if hasattr(res, 'correlation') else np.nan

# --- Main Script Logic ---
def main():
    # 1. Create results directory (relative to project root)
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
        data = data[~background_mask]
        print(f"Filtered {np.sum(background_mask)} background samples. Remaining: {len(data)}")
        if len(data) == 0:
            print("No non-background samples found. Exiting.")
            sys.exit(0)

    # 4. Get unique subjects
    subject_ids = np.unique(data['subject_id'])
    print(f"Found {len(subject_ids)} subjects: {subject_ids.tolist()}")

    # 5. Initialize storage for results
    all_pair_data = []
    subject_correlation_list = []
    cov_estimator = Covariances(estimator='oas')

    # 6. Process each subject
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]
        n_samples = len(subject_data)

        if n_samples != EXPECTED_SAMPLES_PER_SUBJECT:
            print(f"Warning: Subject {subject_id} has {n_samples} samples, expected {EXPECTED_SAMPLES_PER_SUBJECT}. Skipping subject.")
            continue
        # Optional pattern check

        X_subj = subject_data['sample']
        vas_scores_all = subject_data['sample_weight'] # Keep VAS for saving if needed
        covs_subj = cov_estimator.fit_transform(X_subj)

        distances = []
        pair_indices = []
        subject_pairs_temp = []

        # Create pairs and calculate distances
        for i in range(0, n_samples, 2):
            pair_index = i // 2
            cov_rest = covs_subj[i]
            cov_mi = covs_subj[i+1]
            vas_mi = vas_scores_all[i+1] # Get VAS for saving

            dist = distance_riemann(cov_rest, cov_mi)

            distances.append(dist)
            pair_indices.append(pair_index)
            subject_pairs_temp.append({
                'subject_id': subject_id,
                'pair_index': pair_index,
                'distance': dist,
                'vas': vas_mi # Keep VAS score in the detailed data
            })

        all_pair_data.extend(subject_pairs_temp)

        # Calculate Spearman correlation (Distance vs Pair Index) & permutation p-value
        correlation = np.nan
        p_value_spearman = np.nan
        p_value_perm = np.nan

        if len(distances) >= 2:
            distances_arr = np.array(distances)
            pair_indices_arr = np.array(pair_indices)

            if np.std(distances_arr) > 1e-9 and np.std(pair_indices_arr) > 1e-9:
                 correlation, p_value_spearman = spearmanr(distances_arr, pair_indices_arr)
                 perm_result = permutation_test((distances_arr, pair_indices_arr),
                                                _spearman_statistic,
                                                n_resamples=9999,
                                                permutation_type='pairings',
                                                alternative='two-sided',
                                                random_state=RANDOM_STATE)
                 p_value_perm = perm_result.pvalue
            else:
                 print(f"Warning: Could not calculate distance-vs-index correlation for subject {subject_id} due to constant input.")
                 # Assign NaN if correlation cannot be calculated
                 correlation, p_value_spearman, p_value_perm = np.nan, np.nan, np.nan
        else:
            print(f"Warning: Not enough pairs ({len(distances)}) to calculate distance-vs-index correlation for subject {subject_id}.")

        subject_correlation_list.append({
            'subject_id': subject_id,
            'correlation': correlation, # Now Distance vs Pair Index
            'p_value_spearman': p_value_spearman,
            'p_value_perm': p_value_perm,
            'n_pairs': len(distances)
        })

    # 7. Consolidate and Save Results
    if not all_pair_data:
        print("\nNo pair data was generated. Exiting.")
        sys.exit(0)

    overall_pair_df = pd.DataFrame(all_pair_data)
    subject_corr_df = pd.DataFrame(subject_correlation_list)

    # --- Load and Merge F1 Data (Removed Try-Except) ---
    print(f"Loading F1 scores from: {F1_RESULTS_PATH}")
    # Attempt to load directly; FileNotFoundError or other errors will halt execution
    f1_df = pd.read_csv(F1_RESULTS_PATH)
    # Select only necessary columns
    f1_df = f1_df[['subject_id', 'f1']]

    # Merge with correlation data
    merged_metrics_df = pd.merge(subject_corr_df, f1_df, on='subject_id', how='left')
    if merged_metrics_df['f1'].isnull().any():
        # It's possible some subjects exist in one df but not the other after merge,
        # print warning but don't halt unless subsequent code fails.
        print("Warning: Some subjects may be missing F1 scores after merge.")
    # -----------------------------------------------------

    # Save pair data
    pair_data_path = os.path.join(RESULTS_DIR, 'overall_pair_distance_index_data.csv') # Use RESULTS_DIR directly
    overall_pair_df.to_csv(pair_data_path, index=False)
    print(f"\nSaved per-pair distance and index data ({len(overall_pair_df)} rows) to: {pair_data_path}")

    # Save subject correlations (distance vs index)
    subject_corr_path = os.path.join(RESULTS_DIR, 'subject_distance_index_correlation.csv') # Use RESULTS_DIR directly
    # Save the original correlation results, not the merged df, to keep file specific
    subject_corr_df.to_csv(subject_corr_path, index=False)
    print(f"Saved subject distance-vs-index correlation metrics ({len(subject_corr_df)} rows) to: {subject_corr_path}")

    # 8. Generate and Save Plots
    # Plot combined correlation and F1 bar plot
    corr_f1_barplot_path = os.path.join(RESULTS_DIR, 'subject_correlation_f1_barplot.pdf') # Use RESULTS_DIR directly
    plot_correlation_f1_barplot(merged_metrics_df, corr_f1_barplot_path)

    # Keep the original regression plot
    multi_regplot_path = os.path.join(RESULTS_DIR, 'all_subjects_distance_vs_index_regplot.pdf') # Use RESULTS_DIR directly
    plot_all_subjects_regressions(overall_pair_df, multi_regplot_path, n_cols=4)


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main() 