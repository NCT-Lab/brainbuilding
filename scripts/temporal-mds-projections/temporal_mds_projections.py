import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform # For Euclidean distance matrix
import warnings
from tqdm import tqdm
from matplotlib.colors import to_rgb # To convert color names/hex to RGB

# Configure Matplotlib for editable text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy') # Path relative to project root
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = os.path.join('scripts', 'temporal-mds-projections', 'results') # Path relative to project root
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0
RANDOM_STATE = 42 # For MDS reproducibility
N_COMPONENTS_MDS = 2
MIN_ALPHA = 0.2 # Minimum alpha for the earliest point
MAX_ALPHA = 1.0 # Maximum alpha for the latest point

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
    mds_initial = MDS(n_components=N_COMPONENTS_MDS, dissimilarity='precomputed', random_state=RANDOM_STATE,
                      metric=True, n_init=4, max_iter=300, n_jobs=-1)
    mds_final = MDS(n_components=N_COMPONENTS_MDS, dissimilarity='precomputed', random_state=RANDOM_STATE,
                    metric=True, n_init=4, max_iter=300, n_jobs=-1) # Second MDS instance

    # 6. Process each subject and plot
    for i, subject_id in enumerate(tqdm(subject_ids, desc="Processing subjects for Temporal MDS")):
        ax = axes[i]
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]
        n_samples = len(subject_data)

        if n_samples < 2:
            # Handle subjects with insufficient samples
            print(f"Skipping subject {subject_id}: Only {n_samples} sample(s).")
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id}')
            continue

        X_subj = subject_data['sample']
        labels_subj = subject_data['label']
        is_right_hand_subj = subject_data['is_right_hand']

        try:
            # --- Stage 1: Initial MDS based on Riemannian distance ---
            covs_subj = cov_estimator.fit_transform(X_subj)
            dist_riemann_subj = np.zeros((n_samples, n_samples))
            for j in range(n_samples):
                for k in range(j + 1, n_samples):
                    d = distance_riemann(covs_subj[j], covs_subj[k])
                    dist_riemann_subj[j, k] = d
                    dist_riemann_subj[k, j] = d

            mds_coords_2d = mds_initial.fit_transform(dist_riemann_subj)

            # --- Stage 2: Incorporate Time and run second MDS ---
            # Find min/max of initial MDS coordinates
            if mds_coords_2d.size > 0:
                min_val = np.min(mds_coords_2d)
                max_val = np.max(mds_coords_2d)
            else: # Should not happen with n_samples >= 2, but safety first
                min_val, max_val = 0, 1

            # Create and normalize time index
            time_index = np.arange(n_samples)
            if n_samples > 1 and (max_val - min_val) > 1e-9:
                normalized_time_index = (time_index / (n_samples - 1)) * (max_val - min_val) + min_val
            else: # Handle single sample or collapsed initial MDS
                normalized_time_index = np.full(n_samples, min_val)

            # Create 3D features [MDS1, MDS2, NormalizedTime]
            features_3d = np.hstack((mds_coords_2d, normalized_time_index[:, np.newaxis]))

            # Compute pairwise Euclidean distance matrix on 3D features
            dist_euclidean_3d = squareform(pdist(features_3d, metric='euclidean'))

            # Apply final MDS
            mds_coords_final = mds_final.fit_transform(dist_euclidean_3d)
            # --------------------------------------------------------

            # Get color labels and calculate alpha values based on time
            color_labels_subj = get_color_labels(labels_subj, is_right_hand_subj)
            if n_samples > 1:
                # Use original time_index for alpha calculation, not normalized_time_index
                alpha_values = MIN_ALPHA + (time_index / (n_samples - 1)) * (MAX_ALPHA - MIN_ALPHA)
            else:
                alpha_values = np.array([MAX_ALPHA])

            # Create RGBA colors for each point
            point_colors_rgba = []
            for k in range(n_samples):
                label = color_labels_subj[k]
                base_rgb = to_rgb(palette.get(label, 'black')) # Default to black if label not in palette
                alpha = alpha_values[k]
                point_colors_rgba.append((*base_rgb, alpha)) # Append RGBA tuple

            # Plot FINAL MDS projection
            plot_df = pd.DataFrame({
                'MDS1': mds_coords_final[:, 0],
                'MDS2': mds_coords_final[:, 1],
                'label': color_labels_subj
                # Alpha is now encoded in point_colors_rgba
            })
            # Use the generated RGBA colors via 'c', keep style for markers
            sns.scatterplot(data=plot_df, x='MDS1', y='MDS2',
                            c=point_colors_rgba, # Pass the list/array of RGBA colors
                            style='label', markers=markers,
                            ax=ax, s=30,
                            legend=False) # Keep legend manual
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

    fig.suptitle('Per-Subject Temporal MDS Projection', fontsize=16, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    plot_path = os.path.join(RESULTS_DIR, 'all_subjects_temporal_mds_projections.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved multi-subject Temporal MDS plot to: {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main() 