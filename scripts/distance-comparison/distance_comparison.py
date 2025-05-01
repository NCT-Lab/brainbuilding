import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale # Use scikit-learn's minmax_scale
from scipy.spatial.distance import pdist, squareform # For efficient pairwise Euc/Manhattan
import warnings
from tqdm import tqdm

# Configure Matplotlib for editable text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy')
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = os.path.join('scripts', 'distance-comparison', 'results')
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0

# --- Helper Functions ---

def load_data(filepath):
    """Loads structured numpy array from .npy file."""
    absolute_filepath = os.path.abspath(filepath)
    if not os.path.exists(absolute_filepath):
        raise FileNotFoundError(f"Data file not found at {absolute_filepath} (expected relative to project root)")

    data = np.load(absolute_filepath, allow_pickle=True)
    required_fields = ['sample', 'label', 'subject_id', 'is_background'] # Min required fields
    if not all(field in data.dtype.names for field in required_fields):
        raise ValueError(f"Loaded data missing required fields. Found: {data.dtype.names}. Required: {required_fields}")
    if data.size == 0:
        print(f"Warning: Loaded data from {absolute_filepath} is empty.")
    return data

def flatten_covariances(cov_matrices):
    """Flattens the upper triangle (excluding diagonal) of covariance matrices."""
    n_epochs, n_channels, _ = cov_matrices.shape
    # Get indices for the strict upper triangle (k=1 excludes diagonal)
    inds = np.triu_indices(n_channels, k=1)
    n_features = len(inds[0])
    flat_covs_vectors = np.empty((n_epochs, n_features))
    for i, cov in enumerate(cov_matrices):
        flat_covs_vectors[i] = cov[inds]
    return flat_covs_vectors

def plot_r2_scores(metrics_df, output_path):
    """Generates bar plot comparing R2 scores for Euclidean and Manhattan vs Riemannian."""
    if metrics_df.empty or not all(c in metrics_df.columns for c in ['subject_id', 'r2_euclidean', 'r2_manhattan']):
        print("Metrics DataFrame missing required R2 columns, skipping plot.")
        return

    # Sort by Euclidean R2 for consistent ordering (optional)
    sorted_df = metrics_df.sort_values('r2_euclidean', ascending=False).copy()
    sorted_df['subject_id'] = sorted_df['subject_id'].astype(str)
    subject_order = sorted_df['subject_id'].tolist()

    # Melt for plotting with hue
    melted_df = sorted_df.melt(id_vars='subject_id',
                               value_vars=['r2_euclidean', 'r2_manhattan'],
                               var_name='Comparison',
                               value_name='R2 Score')
    # Clean up comparison names for legend
    melted_df['Comparison'] = melted_df['Comparison'].replace({'r2_euclidean': 'vs Euclidean', 'r2_manhattan': 'vs Manhattan'})

    plt.figure(figsize=(14, 7))
    sns.barplot(data=melted_df, x='subject_id', y='R2 Score', hue='Comparison',
                 order=subject_order, palette='Set2')

    plt.title('R2 Score: Riemannian Distance vs. Flattened Distances (Euclidean/Manhattan)')
    plt.xlabel('Subject ID')
    plt.ylabel('R-squared Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Riemannian Compared To')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved R2 comparison plot to: {output_path}")
    plt.close()

def plot_distance_regression(distances_df, output_path, n_cols=4):
    """Generates a multi-subplot figure showing normalized Riemannian vs Euclidean distance."""
    if distances_df.empty or not all(c in distances_df.columns for c in ['subject_id', 'riemann_norm', 'euclidean_norm']):
        print("Distances DataFrame missing required columns, skipping regression plot.")
        return

    subject_ids = sorted(distances_df['subject_id'].unique())
    n_subjects = len(subject_ids)
    n_rows = (n_subjects + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, subject_id in enumerate(tqdm(subject_ids, desc="Generating subject regression plots")):
        ax = axes[i]
        subject_data = distances_df[distances_df['subject_id'] == subject_id]

        if not subject_data.empty:
            sns.regplot(data=subject_data, x='riemann_norm', y='euclidean_norm', ax=ax, lowess=False, # Use linear fit
                        scatter_kws={'s': 5, 'alpha': 0.3}, line_kws={'color': 'red'})
            ax.set_title(f'Subject {subject_id}')
            ax.set_xlabel('Norm. Riemannian Dist.')
            ax.set_ylabel('Norm. Euclidean Dist.')
            # Add R2 text
            r2_val = subject_data['r2_euclidean'].iloc[0] # Get R2 for this subject
            ax.text(0.05, 0.95, f'R2 = {r2_val:.3f}', transform=ax.transAxes, va='top', ha='left', fontsize=9)
            ax.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.5) # Add identity line
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Normalized Riemannian vs. Euclidean Pairwise Distances', fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved multi-subject distance regression plot to: {output_path}")
    plt.close(fig)

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
        data = data[~background_mask]
        print(f"Filtered {np.sum(background_mask)} background samples. Remaining: {len(data)}")
        if len(data) == 0:
            print("No non-background samples found. Exiting.")
            sys.exit(0)

    # 4. Get unique subjects
    subject_ids = np.unique(data['subject_id'])
    print(f"Found {len(subject_ids)} subjects: {subject_ids.tolist()}")

    # 5. Initialize storage
    subject_metrics_list = []
    all_distances_list = [] # To store detailed distances for regression plot
    cov_estimator = Covariances(estimator='oas')

    # 6. Process each subject
    for subject_id in tqdm(subject_ids, desc="Comparing Distances"):
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]
        n_samples = len(subject_data)

        if n_samples < 2:
            print(f"Skipping subject {subject_id}: Only {n_samples} sample(s).")
            continue

        X_subj = subject_data['sample']

        # --- Calculate Distances ---
        try:
            covs_subj = cov_estimator.fit_transform(X_subj)

            # 1. Riemannian
            dist_riemann_matrix = np.zeros((n_samples, n_samples))
            for j in range(n_samples):
                for k in range(j + 1, n_samples):
                    d = distance_riemann(covs_subj[j], covs_subj[k])
                    dist_riemann_matrix[j, k] = d
                    dist_riemann_matrix[k, j] = d
            # Extract unique pairwise distances (vector)
            dist_riemann_vec = dist_riemann_matrix[np.triu_indices(n_samples, k=1)]

            # 2. Euclidean & Manhattan on Flattened Covs
            flat_covs = flatten_covariances(covs_subj)
            dist_euclidean_vec = pdist(flat_covs, metric='euclidean')
            dist_manhattan_vec = pdist(flat_covs, metric='cityblock') # cityblock is Manhattan

            if dist_riemann_vec.size == 0: # Should only happen if n_samples < 2, already checked
                print(f"Warning: No pairwise distances for subject {subject_id}")
                r2_E, r2_M = np.nan, np.nan
                dist_R_norm, dist_E_norm, dist_M_norm = np.array([]), np.array([]), np.array([])
            else:
                # --- Normalize Distances (MinMax) ---
                # Reshape for scaler, then flatten back
                dist_R_norm = minmax_scale(dist_riemann_vec.reshape(-1, 1)).flatten()
                dist_E_norm = minmax_scale(dist_euclidean_vec.reshape(-1, 1)).flatten()
                dist_M_norm = minmax_scale(dist_manhattan_vec.reshape(-1, 1)).flatten()

                # --- Calculate R2 Scores ---
                # Check for constant values in the true distances (Riemannian)
                if np.std(dist_R_norm) > 1e-9:
                    r2_E = r2_score(dist_R_norm, dist_E_norm)
                    r2_M = r2_score(dist_R_norm, dist_M_norm)
                else:
                    print(f"Warning: Normalized Riemannian distances are constant for subject {subject_id}. Cannot calculate R2.")
                    r2_E, r2_M = np.nan, np.nan

            # Store metrics
            subject_metrics_list.append({
                'subject_id': subject_id,
                'r2_euclidean': r2_E,
                'r2_manhattan': r2_M,
                'n_samples': n_samples,
                'n_pairs': len(dist_riemann_vec)
            })
            # Store detailed normalized distances for plotting
            subject_dist_df = pd.DataFrame({
                'subject_id': subject_id,
                'riemann_norm': dist_R_norm,
                'euclidean_norm': dist_E_norm,
                'manhattan_norm': dist_M_norm,
                'r2_euclidean': r2_E # Add R2 here for easy access in plotting
            })
            all_distances_list.append(subject_dist_df)

        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            # Add NaN entry for metrics if error occurs
            subject_metrics_list.append({
                'subject_id': subject_id,
                'r2_euclidean': np.nan,
                'r2_manhattan': np.nan,
                'n_samples': n_samples,
                'n_pairs': 0
            })

    # 7. Consolidate and Save Results
    if not subject_metrics_list:
        print("\nNo subjects processed successfully. Exiting.")
        sys.exit(0)

    subject_metrics_df = pd.DataFrame(subject_metrics_list)
    all_distances_df = pd.concat(all_distances_list, ignore_index=True) if all_distances_list else pd.DataFrame()

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, 'subject_distance_r2_scores.csv')
    subject_metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved R2 score metrics to: {metrics_path}")

    # Optional: Save all normalized distances
    # distances_path = os.path.join(RESULTS_DIR, 'all_normalized_distances.csv')
    # all_distances_df.to_csv(distances_path, index=False)
    # print(f"Saved normalized distances to: {distances_path}")

    # 8. Generate and Save Plots
    r2_plot_path = os.path.join(RESULTS_DIR, 'subject_r2_scores_barplot.pdf')
    plot_r2_scores(subject_metrics_df, r2_plot_path)

    reg_plot_path = os.path.join(RESULTS_DIR, 'all_subjects_riemann_vs_euclidean_regplot.pdf')
    plot_distance_regression(all_distances_df, reg_plot_path)


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main() 