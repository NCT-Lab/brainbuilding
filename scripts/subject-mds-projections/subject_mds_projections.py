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

# Configure Matplotlib for editable text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Constants ---
DEFAULT_DATASET_PATH = os.path.join('data', 'preprocessed', 'motor-imagery-2.npy') # Path relative to project root
DATASET_FNAME = os.getenv("DATASET_FNAME", DEFAULT_DATASET_PATH)
RESULTS_DIR = os.path.join('scripts', 'subject-mds-projections', 'results') # Path relative to project root
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0
RANDOM_STATE = 42 # For MDS reproducibility
N_COMPONENTS_MDS = 2

# --- Helper Functions ---

def load_data(filepath):
    """Loads structured numpy array from .npy file."""
    absolute_filepath = os.path.abspath(filepath)
    if not os.path.exists(absolute_filepath):
        raise FileNotFoundError(f"Data file not found at {absolute_filepath} (expected relative to project root)")

    data = np.load(absolute_filepath, allow_pickle=True)
    # Check for all potentially required fields, including is_background
    required_fields = ['sample', 'label', 'subject_id', 'sample_weight', 'is_background', 'is_right_hand']
    if not all(field in data.dtype.names for field in required_fields):
        raise ValueError(f"Loaded data missing required fields. Found: {data.dtype.names}. Required: {required_fields}")
    if data.size == 0:
        print(f"Warning: Loaded data from {absolute_filepath} is empty.")
    return data

def get_color_labels(labels, is_right_hand, is_background):
    """Assigns string labels for coloring based on sample label, hand, and background status."""
    color_labels = []
    # Ensure is_background is handled correctly, even if not present in all rows (use False as default? Check loading)
    # Assuming load_data ensures is_background is present.
    for label, right_hand, background in zip(labels, is_right_hand, is_background):
        if background:
            color_labels.append('Background')
        elif label == NEGATIVE_CLASS_LABEL:
            color_labels.append('Rest')
        elif label == POSITIVE_CLASS_LABEL:
            if right_hand:
                color_labels.append('Right Hand')
            else:
                color_labels.append('Left Hand')
        else:
            color_labels.append('Other') # Should not happen after filtering
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

    # 3. Filter out background samples - COMMENTED OUT
    # if 'is_background' in data.dtype.names:
    #     background_mask = data['is_background']
    #     data = data[~background_mask]
    #     print(f"Filtered {np.sum(background_mask)} background samples. Remaining: {len(data)}")
    #     if len(data) == 0:
    #         print("No non-background samples found. Exiting.")
    #         sys.exit(0)
    # else:
    #     print("Warning: 'is_background' field not found or handled.") # Add check maybe?

    # 4. Get unique subjects
    subject_ids = np.unique(data['subject_id'])
    n_subjects = len(subject_ids)
    print(f"Found {n_subjects} subjects: {subject_ids.tolist()}")

    # 5. Prepare for plotting
    n_cols = 4
    n_rows = (n_subjects + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    # Define colors and markers - ADD BACKGROUND
    palette = {'Rest': 'grey', 'Right Hand': 'blue', 'Left Hand': 'red', 'Background': 'lightgreen'}
    markers = {'Rest': '.', 'Right Hand': '^', 'Left Hand': 'v', 'Background': 's'} # Square for background

    # Initialize objects for MDS and Covariance estimation
    cov_estimator = Covariances(estimator='oas')
    mds = MDS(n_components=N_COMPONENTS_MDS, dissimilarity='precomputed', random_state=RANDOM_STATE,
              metric=True, n_init=4, max_iter=300, n_jobs=-1)

    # 6. Process each subject and plot
    all_mds_data = [] # Optional: Store MDS results if needed later

    for i, subject_id in enumerate(tqdm(subject_ids, desc="Processing subjects for MDS")):
        ax = axes[i]
        subject_mask = data['subject_id'] == subject_id
        subject_data = data[subject_mask]
        n_samples = len(subject_data)

        if n_samples < 2:
            print(f"Skipping subject {subject_id}: Only {n_samples} sample(s).")
            ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id}')
            continue

        # Extract data for this subject
        X_subj = subject_data['sample']
        labels_subj = subject_data['label']
        is_right_hand_subj = subject_data['is_right_hand']
        is_background_subj = subject_data['is_background'] # Get background flag

        try:
            # Compute Covariances
            covs_subj = cov_estimator.fit_transform(X_subj)

            # Compute Pairwise Riemannian Distances manually
            dist_subj = np.zeros((n_samples, n_samples))
            for j in range(n_samples):
                for k in range(j + 1, n_samples):
                    # Calculate distance between cov j and cov k
                    d = distance_riemann(covs_subj[j], covs_subj[k])
                    dist_subj[j, k] = d
                    dist_subj[k, j] = d # Symmetric matrix

            # Apply MDS (expects precomputed distances)
            mds_coords = mds.fit_transform(dist_subj)

            # Get color labels (now includes background)
            color_labels_subj = get_color_labels(labels_subj, is_right_hand_subj, is_background_subj)

            # Create a DataFrame for plotting convenience
            plot_df = pd.DataFrame({
                'MDS1': mds_coords[:, 0],
                'MDS2': mds_coords[:, 1],
                'label': color_labels_subj
            })
            all_mds_data.append(plot_df.assign(subject_id=subject_id)) # Store if needed

            # Plot MDS projection for the subject
            sns.scatterplot(data=plot_df, x='MDS1', y='MDS2', hue='label', style='label',
                            palette=palette, markers=markers, ax=ax, s=30, alpha=0.8, legend=False)
            ax.set_title(f'Subject {subject_id}')
            ax.set_xticks([])
            ax.set_yticks([])

        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            ax.text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'Subject {subject_id} (Error)')

    # 7. Finalize Plot
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Create a single legend for the whole figure (handles will now include Background)
    handles = [plt.Line2D([0], [0], marker=markers[label], color=palette[label], linestyle='', markersize=6)
               for label in palette.keys()]
    labels = list(palette.keys())
    fig.legend(handles, labels, title='Condition', loc='center right', bbox_to_anchor=(1.0, 0.5))

    fig.suptitle('Per-Subject MDS Projection of Covariance Matrices (Riemannian Distance)', fontsize=16, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legend

    # Save the plot
    plot_path = os.path.join(RESULTS_DIR, 'all_subjects_mds_projections.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Saved multi-subject MDS plot to: {plot_path}")
    plt.close(fig)

    # Optional: Save concatenated MDS data
    # if all_mds_data:
    #     all_mds_df = pd.concat(all_mds_data, ignore_index=True)
    #     mds_data_path = os.path.join(RESULTS_DIR, 'all_subjects_mds_coordinates.csv')
    #     all_mds_df.to_csv(mds_data_path, index=False)
    #     print(f"Saved all MDS coordinates to: {mds_data_path}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    # warnings.filterwarnings('ignore', message='invalid value encountered in sqrtm') # For Riemannian distance if needed
    main() 