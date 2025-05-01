import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from sklearn.manifold import MDS
import warnings
from tqdm import tqdm
from matplotlib.colors import to_rgb

# Configure Matplotlib for editable text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy') # Path relative to project root
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = os.path.join('scripts', 'fused-mds-projections', 'results') # Path relative to project root
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0
RANDOM_STATE = 42 # For MDS reproducibility
N_COMPONENTS_MDS = 2
FUSION_ALPHA = 0.5 # Weight for Riemannian distance
MIN_ALPHA_PLOT = 0.2 # Minimum alpha for plotting the earliest point
MAX_ALPHA_PLOT = 1.0 # Maximum alpha for plotting the latest point

# --- Helper Functions ---

def load_data(filepath):
    """Loads structured numpy array from .npy file."""
    absolute_filepath = os.path.abspath(filepath)
    if not os.path.exists(absolute_filepath):
        raise FileNotFoundError(f"Data file not found at {absolute_filepath} (expected relative to project root)")

    data = np.load(absolute_filepath, allow_pickle=True)
    required_fields = ['sample', 'label', 'subject_id', 'sample_weight', 'is_background', 'is_right_hand']
    if not all(field in data.dtype.names for field in required_fields):
        raise ValueError(f"Loaded data missing required fields. Found: {data.dtype.names}. Required: {required_fields}")
    if data.size == 0:
        print(f"Warning: Loaded data from {absolute_filepath} is empty.")
    return data

def get_color_labels(labels, is_right_hand):
    """Assigns string labels for coloring based on sample label and hand."""
    color_labels = []
    for label, right_hand in zip(labels, is_right_hand):
        if label == NEGATIVE_CLASS_LABEL:
            color_labels.append('Rest')
        elif label == POSITIVE_CLASS_LABEL:
            if right_hand:
                color_labels.append('Right Hand')
            else:
                color_labels.append('Left Hand')
        else:
            color_labels.append('Other')
    return color_labels

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
    n_subjects = len(subject_ids)
    print(f"Found {n_subjects} subjects: {subject_ids.tolist()}")

    # 5. Prepare for plotting
    n_cols = 4
    n_rows = (n_subjects + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    palette = {'Rest': 'grey', 'Right Hand': 'blue', 'Left Hand': 'red'}
    markers = {'Rest': '.', 'Right Hand': '^', 'Left Hand': 'v'}

    # Initialize objects
    cov_estimator = Covariances(estimator='oas')
    mds = MDS(n_components=N_COMPONENTS_MDS, dissimilarity='precomputed', random_state=RANDOM_STATE,
              metric=True, n_init=4, max_iter=300, n_jobs=-1)

    # 6. Process each subject and plot
    for i, subject_id in enumerate(tqdm(subject_ids, desc="Processing subjects for Fused MDS")):
        ax = axes[i]
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]
        n_samples = len(subject_data)

        if n_samples < 2:
            print(f"Skipping subject {subject_id}: Only {n_samples} sample(s).")
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id}')
            continue

        X_subj = subject_data['sample']
        labels_subj = subject_data['label']
        is_right_hand_subj = subject_data['is_right_hand']

        try:
            # --- Calculate Distance Matrices ---
            covs_subj = cov_estimator.fit_transform(X_subj)

            # 1. Riemannian Distance Matrix
            dist_riemann_subj = np.zeros((n_samples, n_samples))
            for j in range(n_samples):
                for k in range(j + 1, n_samples):
                    d = distance_riemann(covs_subj[j], covs_subj[k])
                    dist_riemann_subj[j, k] = d
                    dist_riemann_subj[k, j] = d

            # 2. Temporal Distance Matrix
            time_indices = np.arange(n_samples).reshape(-1, 1)
            dist_time_subj = np.abs(time_indices - time_indices.T)

            # --- Scale Temporal Distance Matrix ---
            # Get range of off-diagonal Riemannian distances
            mask_offdiag = ~np.eye(n_samples, dtype=bool)
            if np.any(mask_offdiag):
                riemann_vals_offdiag = dist_riemann_subj[mask_offdiag]
                min_R = np.min(riemann_vals_offdiag) if riemann_vals_offdiag.size > 0 else 0
                max_R = np.max(riemann_vals_offdiag) if riemann_vals_offdiag.size > 0 else 1
            else: # Only one unique point (or n_samples=1, though filtered earlier)
                 min_R, max_R = 0, 1

            # Get range of off-diagonal Temporal distances
            if np.any(mask_offdiag):
                time_vals_offdiag = dist_time_subj[mask_offdiag]
                min_T = np.min(time_vals_offdiag) if time_vals_offdiag.size > 0 else 0 # Usually 1
                max_T = np.max(time_vals_offdiag) if time_vals_offdiag.size > 0 else 1 # Usually n_samples - 1
            else:
                min_T, max_T = 0, 1

            # Scale temporal distances to Riemannian range
            scaled_dist_time_subj = np.zeros_like(dist_time_subj, dtype=float)
            range_T = max_T - min_T
            range_R = max_R - min_R
            if range_T > 1e-9: # Avoid division by zero if all time distances are the same (n_samples=2)
                scaled_dist_time_subj[mask_offdiag] = (((dist_time_subj[mask_offdiag] - min_T) / range_T) * range_R) + min_R
            else: # If only one unique time distance, map it to the middle of R range or just min_R
                 scaled_dist_time_subj[mask_offdiag] = min_R + range_R / 2.0

            # Ensure diagonal remains zero
            np.fill_diagonal(scaled_dist_time_subj, 0)

            # --- Fuse Distance Matrices ---
            dist_fused_subj = FUSION_ALPHA * dist_riemann_subj + (1 - FUSION_ALPHA) * scaled_dist_time_subj

            # --- Apply MDS --- #
            mds_coords = mds.fit_transform(dist_fused_subj)

            # --- Plotting --- #
            color_labels_subj = get_color_labels(labels_subj, is_right_hand_subj)
            time_index_orig = np.arange(n_samples) # For alpha calculation
            if n_samples > 1:
                alpha_values = MIN_ALPHA_PLOT + (time_index_orig / (n_samples - 1)) * (MAX_ALPHA_PLOT - MIN_ALPHA_PLOT)
            else:
                alpha_values = np.array([MAX_ALPHA_PLOT])

            point_colors_rgba = []
            for k in range(n_samples):
                label = color_labels_subj[k]
                base_rgb = to_rgb(palette.get(label, 'black'))
                alpha = alpha_values[k]
                point_colors_rgba.append((*base_rgb, alpha))

            plot_df = pd.DataFrame({
                'MDS1': mds_coords[:, 0],
                'MDS2': mds_coords[:, 1],
                'label': color_labels_subj
            })
            sns.scatterplot(data=plot_df, x='MDS1', y='MDS2',
                            c=point_colors_rgba,
                            style='label', markers=markers,
                            ax=ax, s=30,
                            legend=False)
            ax.set_title(f'Subject {subject_id}')
            ax.set_xticks([])
            ax.set_yticks([])

        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id} (Error)')

    # 7. Finalize Plot
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Line2D([0], [0], marker=markers[label], color=palette[label], linestyle='', markersize=6)
               for label in palette.keys()]
    labels = list(palette.keys())
    fig.legend(handles, labels, title='Condition', loc='center right', bbox_to_anchor=(1.0, 0.5))

    fig.suptitle('Per-Subject MDS Projection (Fused: 0.5*Riemann + 0.5*ScaledTime)', fontsize=16, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    plot_path = os.path.join(RESULTS_DIR, 'all_subjects_fused_mds_projections.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved multi-subject Fused MDS plot to: {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main() 