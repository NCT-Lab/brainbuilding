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
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy')
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = 'results'
POSITIVE_CLASS_LABEL = 1 # Label designated for motor imagery
NEGATIVE_CLASS_LABEL = 0 # Label designated for rest/baseline
EXPECTED_SAMPLES_PER_SUBJECT = 120
EXPECTED_PAIRS_PER_SUBJECT = EXPECTED_SAMPLES_PER_SUBJECT // 2
RANDOM_STATE = 42 # For reproducibility (though not used much here)

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

def plot_correlation_barplot(metrics_df, output_path):
    """Generates and saves a bar plot of Spearman correlation per subject, sorted,
       with significance asterisks based on permutation p-values."""
    if metrics_df.empty or 'correlation' not in metrics_df.columns or 'p_value_perm' not in metrics_df.columns:
        print("Metrics DataFrame is empty or missing 'correlation'/'p_value_perm', skipping bar plot.")
        return

    sorted_df = metrics_df.sort_values('correlation', ascending=False).copy()
    sorted_df['subject_id'] = sorted_df['subject_id'].astype(str)
    subject_order = sorted_df['subject_id'].tolist()

    plt.figure(figsize=(14, 8)) # Slightly wider figsize
    ax = sns.barplot(data=sorted_df, x='subject_id', y='correlation', order=subject_order, palette='viridis')

    plt.title(f'Per-Subject Spearman Correlation (Distance vs. VAS) with Permutation Test Significance')
    plt.xlabel('Subject ID')
    plt.ylabel('Spearman Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')

    # Add significance asterisks
    # Get heights and positions of bars
    heights = [p.get_height() for p in ax.patches]
    positions = [p.get_x() + p.get_width() / 2 for p in ax.patches]

    # Create mapping from subject_id (string) to p_value
    p_values_map = sorted_df.set_index('subject_id')['p_value_perm'].to_dict()

    y_max = plt.ylim()[1] # Get current y-axis limit
    offset = y_max * 0.02 # Offset for placing asterisks

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
                sig_symbol = '' # No symbol for p >= 0.05

            if sig_symbol:
                bar_height = heights[i]
                # Adjust vertical position slightly above the bar
                # Handle negative bars: place below zero
                text_y = bar_height + offset if bar_height >= 0 else bar_height - offset * 2 # Lower offset for neg bars
                va = 'bottom' if bar_height >= 0 else 'top'
                ax.text(positions[i], text_y, sig_symbol, ha='center', va=va, color='black', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved correlation bar plot with significance to: {output_path}")
    plt.close()

def plot_all_subjects_regressions(pair_data_df, output_path, n_cols=4):
    """Generates a multi-subplot figure showing distance vs VAS regression for each subject,
       with points colored by pair index."""
    if pair_data_df.empty or not all(c in pair_data_df.columns for c in ['subject_id', 'distance', 'vas', 'pair_index']):
        print("Pair data DataFrame is empty or missing required columns (incl. 'pair_index'), skipping multi-subject regression plot.")
        return

    subject_ids = sorted(pair_data_df['subject_id'].unique())
    n_subjects = len(subject_ids)
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4.5), sharex=False, sharey=False) # Slightly larger figure
    axes = axes.flatten() # Flatten to easily iterate

    # Find global min/max pair index for consistent color mapping
    norm = plt.Normalize(pair_data_df['pair_index'].min(), pair_data_df['pair_index'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([]) # Needed for colorbar

    for i, subject_id in enumerate(tqdm(subject_ids, desc="Generating subject regression plots")):
        ax = axes[i]
        subject_data = pair_data_df[pair_data_df['subject_id'] == subject_id].copy()
        # Ensure pair_index is numeric
        subject_data['pair_index'] = pd.to_numeric(subject_data['pair_index'])

        if len(subject_data) > 1:
            # 1. Plot the LOESS regression line only
            sns.regplot(data=subject_data, x='distance', y='vas', ax=ax, lowess=True,
                        scatter=False, # Turn off scatter plotting in regplot
                        line_kws={'color': 'black', 'lw': 1.5})

            # 2. Plot the scatter points separately, colored by pair_index
            sns.scatterplot(data=subject_data, x='distance', y='vas', ax=ax,
                            hue='pair_index', palette='coolwarm', norm=norm, # Use hue and palette
                            s=25, alpha=0.7, legend=False) # No legend needed, use colorbar

            ax.set_title(f'Subject {subject_id}')
            ax.set_xlabel('Riemannian Distance')
            ax.set_ylabel('VAS Score')
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id}')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Distance vs. VAS (Points Colored by Pair Order)', fontsize=16, y=1.03) # Adjusted title & y
    # Add a single colorbar for the pair index
    fig.colorbar(sm, ax=axes[:n_subjects], orientation='vertical', fraction=0.01, pad=0.02, label='Pair Index')
    fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout slightly for suptitle and colorbar
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved multi-subject regression plot with hue to: {output_path}")
    plt.close(fig)


# --- Main Script Logic ---

def _spearman_statistic(x, y):
    """Helper function to calculate Spearman correlation for permutation test."""
    res = spearmanr(x, y)
    # Handle potential NaN result if input is constant
    return res.correlation if hasattr(res, 'correlation') else np.nan

def main():
    # 1. Create results directory relative to the script
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
    all_pair_data = []
    subject_correlation_list = []
    cov_estimator = Covariances(estimator='oas')

    # 6. Process each subject
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]
        n_samples = len(subject_data)

        # Verify sample count and pattern (optional but recommended)
        if n_samples != EXPECTED_SAMPLES_PER_SUBJECT:
            print(f"Warning: Subject {subject_id} has {n_samples} samples, expected {EXPECTED_SAMPLES_PER_SUBJECT}. Skipping subject.")
            continue
        # Optional: Check alternating pattern 0, 1, 0, 1...
        # labels = subject_data['label']
        # if not np.all(labels[::2] == NEGATIVE_CLASS_LABEL) or not np.all(labels[1::2] == POSITIVE_CLASS_LABEL):
        #     print(f"Warning: Subject {subject_id} does not follow expected 0, 1 pattern. Skipping subject.")
        #     continue

        # Extract EEG data and VAS scores
        X_subj = subject_data['sample']
        vas_scores_all = subject_data['sample_weight']

        # Compute all covariance matrices for the subject
        covs_subj = cov_estimator.fit_transform(X_subj)

        distances = []
        vas_paired = []
        subject_pairs_temp = []

        # Create pairs and calculate distances
        for i in range(0, n_samples, 2):
            pair_index = i // 2
            cov_rest = covs_subj[i]    # Covariance for sample 0, 2, 4...
            cov_mi = covs_subj[i+1]  # Covariance for sample 1, 3, 5...
            vas_mi = vas_scores_all[i+1] # VAS for sample 1, 3, 5...

            # Calculate Riemannian distance between the pair
            # distance_riemann expects shape (n_matrices, n_channels, n_channels)
            # We calculate one distance at a time here
            dist = distance_riemann(cov_rest, cov_mi)

            distances.append(dist)
            vas_paired.append(vas_mi)
            subject_pairs_temp.append({
                'subject_id': subject_id,
                'pair_index': pair_index,
                'distance': dist,
                'vas': vas_mi
            })

        all_pair_data.extend(subject_pairs_temp)

        # Calculate Spearman correlation for the subject
        correlation = np.nan
        p_value_spearman = np.nan # Original Spearman p-value (optional to keep)
        p_value_perm = np.nan # Permutation test p-value

        if len(distances) >= 2:
            distances_arr = np.array(distances)
            vas_paired_arr = np.array(vas_paired)

            # Check for constant values which would make correlation undefined
            if np.std(distances_arr) > 1e-9 and np.std(vas_paired_arr) > 1e-9:
                 try:
                     correlation, p_value_spearman = spearmanr(distances_arr, vas_paired_arr)

                     # Run permutation test
                     perm_result = permutation_test((distances_arr, vas_paired_arr),
                                                    _spearman_statistic,
                                                    n_resamples=9999, # Number of permutations
                                                    permutation_type='pairings', # Shuffle VAS scores relative to distances
                                                    alternative='two-sided', # Check for positive or negative correlation
                                                    random_state=RANDOM_STATE)
                     p_value_perm = perm_result.pvalue

                 except ValueError as e:
                      print(f"Error calculating correlation/permutation for subject {subject_id}: {e}")
                      correlation, p_value_spearman, p_value_perm = np.nan, np.nan, np.nan
            else:
                 print(f"Warning: Could not calculate correlation for subject {subject_id} due to constant distances or VAS scores.")
        else:
            print(f"Warning: Not enough pairs ({len(distances)}) to calculate correlation for subject {subject_id}.")

        subject_correlation_list.append({
            'subject_id': subject_id,
            'correlation': correlation,
            'p_value_spearman': p_value_spearman, # Keep original p-value if desired
            'p_value_perm': p_value_perm,       # Add permutation p-value
            'n_pairs': len(distances)
        })

    # 7. Consolidate and Save Results
    if not all_pair_data:
        print("\nNo pair data was generated. Exiting.")
        sys.exit(0)

    overall_pair_df = pd.DataFrame(all_pair_data)
    subject_corr_df = pd.DataFrame(subject_correlation_list)

    # Save pair data (already contains pair_index)
    pair_data_path = os.path.join(results_full_path, 'overall_pair_data.csv')
    overall_pair_df.to_csv(pair_data_path, index=False)
    print(f"\nSaved per-pair distance and VAS data ({len(overall_pair_df)} rows) to: {pair_data_path}")

    # Save subject correlations
    subject_corr_path = os.path.join(results_full_path, 'subject_distance_vas_correlation.csv')
    subject_corr_df.to_csv(subject_corr_path, index=False)
    print(f"Saved subject correlation metrics ({len(subject_corr_df)} rows) to: {subject_corr_path}")

    # 8. Generate and Save Plots
    corr_barplot_path = os.path.join(results_full_path, 'subject_correlation_barplot.pdf')
    plot_correlation_barplot(subject_corr_df, corr_barplot_path)

    multi_regplot_path = os.path.join(results_full_path, 'all_subjects_distance_vs_vas_regplot.pdf')
    plot_all_subjects_regressions(overall_pair_df, multi_regplot_path, n_cols=4)


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # Ignore specific RuntimeWarning from distance calculation if needed (e.g., from sqrtm)
    # warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in sqrtm')
    main() 