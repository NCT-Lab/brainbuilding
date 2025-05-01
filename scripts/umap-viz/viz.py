import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Scaler, Filtering
from gtda.diagrams import PersistenceLandscape
from gtda.plotting import plot_diagram
from gtda.diagrams import PairwiseDistance
from sklearn.preprocessing import FunctionTransformer
from brainbuilding.transformers import (
    SubjectWhiteningTransformer, 
    BackgroundFilterTransformer, 
    Covariances, 
    ParallelTransportTransformer, 
    ColumnSelector, 
    StructuredCSP,
    StructuredColumnTransformer,
    TangentSpaceProjector,
    AugmentedDataset
)

# Configuration
DATASET_FNAME = os.getenv("DATASET_NAME", 'data/preprocessed/motor-imagery-2.npy')
N_FILTERS = 10
OUTPUT_DIR = "output/umap-viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load preprocessed data from numpy file"""
    print(f"Loading data from {DATASET_FNAME}")
    return np.load(DATASET_FNAME, allow_pickle=True)

# Function to reshape the 3D output of PersistenceLandscape to 2D for the classifier
def reshape_features(X):
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X # Return as is if already 2D

def create_pipeline():
    """Create pipeline for data preprocessing before UMAP"""
    return Pipeline([
        ('whitening', SubjectWhiteningTransformer()),
        ('background_filter', BackgroundFilterTransformer()),
        # ('augmented', StructuredColumnTransformer(column='sample', transformer=AugmentedDataset(order=4, lag=8))),
        ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
        ('pt', ParallelTransportTransformer(include_means=True)),
        ('select', ColumnSelector(fields='sample')),
        ('VR', VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0, 1, 2])),
        # ('scaler', Scaler()),
        # ('filter', Filtering()),
        ("filtration", Filtering(homology_dimensions=[0, 1, 2])),
        # ("scale", Scaler()),
        # ('vectorization', PersistenceLandscape(
        #     n_layers=4,
        #     n_bins=50,
        #     n_jobs=-1)),
        # ('reshape', FunctionTransformer(reshape_features, validate=False))
        # ('entropy', PersistenceEntropy()),
        # ('tangent', TangentSpaceProjector(sample_col='sample', general_mean_col='general_mean')),
        # ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=True, metric='riemann'))
    ])

def apply_umap(data, n_components=2, n_neighbors=720, metric='euclidean'):
    """Apply UMAP dimensionality reduction with adjusted parameters for better separation"""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        # min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_epochs=100,
        verbose=True
    )
    return reducer.fit_transform(data)

def main():
    # Load data
    data = load_data()
    
    # Create and apply pipeline
    print("Applying preprocessing pipeline...")
    pipeline = create_pipeline()
    
    # Fit the pipeline on non-background data
    print("Fitting pipeline...")
    pipeline.fit(data, data['label'][data['is_background'] == 0])
    data = data[data['is_background'] == 0]
    
    # Transform the data (output is the distance matrix)
    print("Transforming data (computing distance matrix)...")
    X_dist = pipeline.transform(data)
    print(f"Computed distance matrix shape: {X_dist.shape}")
    
    # Prepare data for UMAP - X is now the distance matrix
    # The data array (already filtered) is used for labels
    
    # Create labels (using the already filtered data)
    labels = np.full(len(data), -1, dtype=int)  # Initialize with -1 to catch unlabeled samples
    
    # Label categories:
    # 0: Rest
    # 1: Right hand point
    # 2: Right hand image
    # 3: Left hand point
    # 4: Left hand image
    
    # Rest samples: label=0 and not background
    is_rest = (data['label'] == 0)
    labels[is_rest] = 0
    
    # Imagery samples: label=1
    is_imagery = data['label'] == 1
    
    # Right hand imagery samples
    is_right_hand = data['is_right_hand'] & is_imagery
    
    # Right hand point
    is_right_hand_point = is_right_hand & data['is_point']
    labels[is_right_hand_point] = 1
    
    # Right hand image (not point)
    is_right_hand_image = is_right_hand & ~data['is_point']
    labels[is_right_hand_image] = 2
    
    # Left hand
    is_left_hand = ~data['is_right_hand'] & is_imagery
    
    # Left hand point
    is_left_hand_point = is_left_hand & data['is_point']
    labels[is_left_hand_point] = 3
    
    # Left hand image
    is_left_hand_image = is_left_hand & ~data['is_point']
    labels[is_left_hand_image] = 4
    
    # Check if we have any unlabeled samples
    unlabeled = labels == -1
    if np.any(unlabeled):
        print(f"Warning: {np.sum(unlabeled)} samples were not labeled!")
    
    # Count samples per class
    for i, name in enumerate(['Rest', 'Right hand point', 'Right hand image', 'Left hand point', 'Left hand image']):
        count = np.sum(labels == i)
        print(f"Class {i} ({name}): {count} samples")
    
    # Apply UMAP using the precomputed distance matrix
    print("Applying UMAP...")
    embedding = apply_umap(X_dist, metric='euclidean')
    
    # Define better color scheme with stronger contrast
    colors = [
        '#3366CC',  # Rest - Blue
        '#FF9900',  # Right hand point - Orange
        '#FFCC99',  # Right hand image - Light orange
        '#33CC33',  # Left hand point - Green
        '#99FF99'   # Left hand image - Light green
    ]
    
    # Create plot with seaborn styling for better aesthetics
    sns.set(style="whitegrid", context="notebook")
    plt.figure(figsize=(14, 12))
    
    # Plot each class
    class_names = ['Rest', 'Right hand point', 'Right hand image', 'Left hand point', 'Left hand image']
    
    for i, name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=colors[i],
                label=f"{name} (n={np.sum(mask)})",
                alpha=0.8,
                s=80,  # Slightly larger point size
                edgecolor='w',  # White edge for better visibility
                linewidth=0.5
            )
    
    plt.legend(fontsize=14, markerscale=1.5, title="Classes", title_fontsize=16)
    plt.title('UMAP Visualization of Motor Imagery EEG Data', fontsize=20, pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=16, labelpad=15)
    plt.ylabel('UMAP Dimension 2', fontsize=16, labelpad=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save figure with higher resolution
    output_path = os.path.join(OUTPUT_DIR, "umap_visualization.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    plt.show()
    
    plt.close()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=data['subject_id'],
        label="Subject ID",
        alpha=0.8,
        s=80,  # Slightly larger point size
        edgecolor='w',  # White edge for better visibility
        cmap='tab10', # discrete colors
        linewidth=0.5
    )
    
    
    # Show figure
    plt.show()

if __name__ == "__main__":
    main()
