import numpy as np
import os
import math
from sklearn.pipeline import Pipeline
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from brainbuilding.transformers import (
    SubjectWhiteningTransformer,
    BackgroundFilterTransformer,
    Covariances,
    ParallelTransportTransformer,
    ColumnSelector,
    StructuredColumnTransformer,
    # AugmentedDataset # Removed - not used
    # TangentSpaceProjecter # Removed - not used
    # StructuredCSP # Removed - not used
)
from pyriemann.utils.distance import distance_riemann

# Configuration
DATASET_FNAME = os.getenv("DATASET_NAME", 'data/preprocessed/motor-imagery-2.npy')
OUTPUT_DIR = "output/persistence-diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load preprocessed data from numpy file"""
    print(f"Loading data from {DATASET_FNAME}")
    data = np.load(DATASET_FNAME, allow_pickle=True)
    # Ensure required columns exist
    required_cols = ['sample', 'label', 'is_background', 'subject_id', 'is_right_hand', 'is_point']
    if not all(col in data.dtype.names for col in required_cols):
        raise ValueError(f"Input data must contain columns: {required_cols}")
    return data

def riemann_distance_wrapper(vec1, vec2, n_channels):
    """
    Custom distance function for VietorisRipsPersistence.
    Reshapes flattened vectors into square matrices and computes the
    Riemannian distance between them.
    """
    mat1 = vec1.reshape(n_channels, n_channels)
    mat2 = vec2.reshape(n_channels, n_channels)

    # Compute the Riemannian distance between the two matrices
    return distance_riemann(mat1, mat2)


def create_simple_cov_pipeline():
    """Pipeline for calculating simple covariances."""
    return Pipeline([
        ('background_filter', BackgroundFilterTransformer()),
        ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
        ('select', ColumnSelector(fields='sample')), # Select only the computed covariances
    ])

def create_whitened_cov_pt_pipeline():
    """Pipeline for whitening, covariances, and parallel transport."""
    return Pipeline([
        # Whitening depends on subject_id and potentially labels for grouping
        ('whitening', SubjectWhiteningTransformer()),
        ('background_filter', BackgroundFilterTransformer()),
        # Covariances applied to whitened data
        ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
        # Parallel transport needs reference points, often computed per class (needs labels)
        ('pt', ParallelTransportTransformer(include_means=True)), # include_means=True might change output structure, check transformer details
        ('select', ColumnSelector(fields='sample')), # Select the transported covariances
    ])


def process_and_plot_diagram(pipeline, data, pipeline_name, output_dir):
    """
    Fits the pipeline, transforms data, filters background, calculates
    the persistence diagram using VietorisRipsPersistence with Riemannian metric,
    and plots the result.
    """
    print(f"\n--- Processing data for pipeline: {pipeline_name} ---")

    # Determine if the pipeline requires labels ('y') for fitting
    # Whitening and PT typically require subject/label info provided via structured array access or y
    needs_y_fit = isinstance(pipeline.steps[0][1], SubjectWhiteningTransformer) or \
                  any(isinstance(step, ParallelTransportTransformer) for _, step in pipeline.steps)

    print(f"Fitting pipeline {pipeline_name}...")
    if needs_y_fit:
        # Pass the full structured array and labels if needed by transformers internally
        # Assuming transformers are designed to handle structured arrays or select necessary columns
        pipeline.fit(data, data['label'])
    else:
        pipeline.fit(data)

    # Transform data
    print(f"Transforming data with {pipeline_name}...")
    # The pipeline output should be an array of SPD matrices
    X_transformed = pipeline.transform(data) # Shape: (n_total_samples, n_channels, n_channels)
    print(f"Transformed data shape (all samples): {X_transformed.shape}")

    n_samples_filt = X_transformed.shape[0]
    print(f"Filtered data shape (non-background): {X_transformed.shape}") # Shape: (n_filtered_samples, n_channels, n_channels)

    if X_transformed.ndim != 3 or X_transformed.shape[1] != X_transformed.shape[2]:
         raise ValueError(f"Pipeline {pipeline_name} did not produce expected matrix array shape. Got {X_transformed.shape}")

    # Get matrix dimensions (n_channels x n_channels)
    _, n_channels, _ = X_transformed.shape
    n_features_flat = n_channels * n_channels

    # Flatten matrices and reshape for VietorisRipsPersistence input format
    # VR expects (n_samples, n_features) or (n_samples, n_points, n_dims)
    # We treat each SPD matrix as a 'point' and compute pairwise distances.
    # Input shape should be (1, n_filtered_samples, n_flattened_features)
    # where n_flattened_features = n_channels * n_channels
    print("Flattening and reshaping matrices for Vietoris-Rips...")
    X_flat = X_transformed.reshape(n_samples_filt, n_features_flat)
    X_vr_input = X_flat[np.newaxis, :, :] # Add batch dimension: (1, n_samples_filt, n_features_flat)
    print(f"Input shape for VietorisRipsPersistence: {X_vr_input.shape}")

    # Define metric parameters for the custom distance function
    metric_params = {'n_channels': n_channels}

    # Calculate persistence diagram using Vietoris-Rips
    print("Calculating persistence diagram using Vietoris-Rips complex...")
    vr = VietorisRipsPersistence(
        metric=riemann_distance_wrapper, # Use our custom Riemann distance
        metric_params=metric_params,    # Pass n_channels to the wrapper
        homology_dimensions=[0, 1, 2],     # Compute H0 (connected components) and H1 (loops)
        n_jobs=-1,                      # Use all available CPU cores
    )

    # fit_transform expects input shape (n_samples, n_points, n_dimensions)
    # Here, n_samples=1, n_points=n_samples_filt, n_dimensions=n_features_flat
    diagram = vr.fit_transform(X_vr_input) # Output shape (n_samples, n_intervals, 3) -> (1, n_intervals, 3)
    diagram = diagram[0] # Remove the first dimension, leaving (n_intervals, 3)
    print(f"Computed diagram shape: {diagram.shape}")

    # Plot the diagram using gtda.plotting
    print("Plotting diagram...")
    # plot_diagram expects shape (n_points, 3)
    if diagram.shape[0] > 0: # Only plot if there are points in the diagram
        fig = plot_diagram(diagram)
        plot_filename = os.path.join(output_dir, f"persistence_diagram_{pipeline_name}.html")
        fig.write_html(plot_filename)
        print(f"Persistence diagram saved to {plot_filename}")
    else:
        print("Diagram is empty, skipping plot.")

    return diagram

# Main function
def main():
    # Load data (structured numpy array)
    data = load_data()

    # Create the two pipelines
    pipe_simple_cov = create_simple_cov_pipeline()
    pipe_whitened_pt = create_whitened_cov_pt_pipeline()

    # --- Process and plot for simple covariance pipeline ---
    diagram_simple = process_and_plot_diagram(
        pipeline=pipe_simple_cov,
        data=data,
        pipeline_name="simple_covariance",
        output_dir=OUTPUT_DIR
    )

    # --- Process and plot for whitened + PT covariance pipeline ---
    diagram_whitened = process_and_plot_diagram(
        pipeline=pipe_whitened_pt,
        data=data,
        pipeline_name="whitened_pt_covariance",
        output_dir=OUTPUT_DIR
    )

    # --- Summary ---
    print("\n--- Processing Complete ---")
    if diagram_simple is not None:
        print(f"Simple covariance diagram: Found {diagram_simple.shape[0]} persistence pairs.")
    else:
        print("Simple covariance diagram: Not computed (no non-background samples or error).")

    if diagram_whitened is not None:
        print(f"Whitened+PT covariance diagram: Found {diagram_whitened.shape[0]} persistence pairs.")
    else:
        print("Whitened+PT covariance diagram: Not computed (no non-background samples or error).")

    print(f"Output plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
