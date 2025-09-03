import numpy as np
import os
import math
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.diagrams import PersistenceLandscape
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from functools import partial
from gtda.diagrams import PairwiseDistance, Filtering, Scaler
from brainbuilding.core.transformers import (
    BackgroundFilterTransformer,
    Covariances,
    StructuredColumnTransformer,
    ColumnSelector,
    SubjectWhiteningTransformer,
    AugmentedDataset,
)
from sklearn.base import BaseEstimator, TransformerMixin
from tslearn.metrics import dtw
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from fastdtw import fastdtw

# Configuration
DATASET_FNAME = os.getenv("DATASET_NAME", 'data/preprocessed/motor-imagery-2.npy')
OUTPUT_DIR = "output/subject-knn-eval-persistence"
os.makedirs(OUTPUT_DIR, exist_ok=True)
KNN_N_NEIGHBORS_LANDSCAPE = 25
LANDSCAPE_N_BINS = 50
HOMOLOGY_DIMENSIONS = (0, 1, 2, 3)


# --- Data Loading ---
def load_data():
    """Load preprocessed data from numpy file"""
    print(f"Loading data from {DATASET_FNAME}")
    data = np.load(DATASET_FNAME, allow_pickle=True)
    print(data['sample'].shape)
    required_cols = ['sample', 'label', 'is_background', 'subject_id', 'is_right_hand', 'is_point', 'event_id']
    if not all(col in data.dtype.names for col in required_cols):
        raise ValueError(f"Input data must contain columns: {required_cols}")
    return data

# --- Helper Functions for Pipeline Steps ---
def covariance_to_correlation_distance(cov_matrix):
    return np.sqrt(2*(1 - cov_matrix))

# --- Custom Per-Subject Scaler ---
class PerSubjectMinMaxScaler(BaseEstimator, TransformerMixin):
    """Applies Min-Max scaling to the 'sample' field of a structured array, per subject."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        unique_subjects = np.unique(X['subject_id'])

        for subject_id in unique_subjects:
            subject_mask = (X['subject_id'] == subject_id)
            subject_samples = X['sample'][subject_mask]

            min_val = np.min(subject_samples)
            max_val = np.max(subject_samples)
            denominator = max_val - min_val
            X_transformed['sample'][subject_mask] = (subject_samples - min_val) / denominator

        return X_transformed

# --- Main Execution Logic ---
def main():
    # 1. Load Data
    print("--- Step 1: Load Data ---")
    data = load_data()

    # 2. Create and Run Feature Extraction Pipeline (Data -> Landscapes)
    print("\n--- Step 2: Run Landscape Pipeline ---")

    full_pipeline = Pipeline([
        # ('whitening', SubjectWhiteningTransformer()),
        ('background_filter', BackgroundFilterTransformer()),
        # ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
        # ('reshape', FunctionTransformer(lambda x: x.reshape(x.shape[0], -1))),
        # ('per_subject_scaler', PerSubjectMinMaxScaler()),
        ('selector', ColumnSelector(fields='sample')),
        ("augmentation", AugmentedDataset(order=4, lag=8)),
        # ("dtw_dist", FunctionTransformer(lambda x: np.array([pairwise_distances(i, metric=dtw, n_jobs=-1) for i in tqdm(x, desc="Computing DTW Distances")]))),
        # ('scaler', FunctionTransformer(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))),
        ('transpose', FunctionTransformer(lambda x: np.array([i.T for i in x]))),
        
        # ('scaler', FunctionTransformer(lambda x: x + np.min(x))),
        # ('cov_to_dist', FunctionTransformer(lambda x: 1 - x)),
        # ('cov_to_dist', FunctionTransformer(covariance_to_correlation_distance)),
        ("VR", VietorisRipsPersistence(metric="cosine", homology_dimensions=HOMOLOGY_DIMENSIONS, n_jobs=-1)),
        ("scaler", Scaler()),
        ('filter', Filtering(epsilon=0.1)),
        # ("landscape", PersistenceLandscape(n_bins=LANDSCAPE_N_BINS)),

    ])
    data = data[data['subject_id'] == 14]
    processed_data = full_pipeline.fit_transform(data)
    print(processed_data.shape)
    print(np.max(processed_data), np.min(processed_data))
    # sample = processed_data[0:1]
    # VR = VietorisRipsPersistence(metric="euclidean", homology_dimensions=HOMOLOGY_DIMENSIONS, n_jobs=-1)
    # sample_transformed = VR.fit_transform(sample)
    # filtration = Filtering(epsilon=2e-6)
    # sample_transformed = filtration.fit_transform(sample_transformed)
    # fig = plot_diagram(sample_transformed[0])
    # plot_filename = os.path.join(OUTPUT_DIR, f"persistence_diagram.html")
    # fig.write_html(plot_filename)
    # print(sample_transformed.shape)
    # exit()
    # distance_matrix = pairwise_distances(processed_data[0], metric=dtw)
    # plt.imshow(processed_data[0])
    # plt.show()
    # exit()
    # exit()
    data = data[data['is_background'] == 0]
    y = data['label']
    subject_ids = data['subject_id']
    for subject_id in np.unique(subject_ids):
        subject_data = processed_data[data['subject_id'] == subject_id]
        # subject_data = subject_data.reshape(subject_data.shape[0], -1)
        distance_matrix = PairwiseDistance(metric='landscape').fit_transform(subject_data)
        # distance_matrix = pairwise_distances(metric).fit_transform(subject_data)
        # finish this
        subject_labels = y[data['subject_id'] == subject_id]
        n_samples = subject_data.shape[0]

        if n_samples < 2:
            print(f"Subject {subject_id}: Skipping evaluation (only {n_samples} samples).")
            continue

        effective_k = min(KNN_N_NEIGHBORS_LANDSCAPE, n_samples - 1)
        if effective_k <= 0:
            print(f"Subject {subject_id}: Skipping evaluation (cannot find neighbors, n_samples={n_samples}).")
            continue

        print(f"Subject {subject_id}: Running LOOCV with k={effective_k}...")

        knn = KNeighborsClassifier(n_neighbors=effective_k, metric='precomputed')
        loo = LeaveOneOut()

        # Use cross_val_predict to get predictions for each sample when it's in the test set
        # Note: Requires the full subject distance matrix as 'X'
        y_pred = cross_val_predict(knn, distance_matrix, subject_labels, cv=loo)
        y_true = subject_labels # True labels are just the subject's labels

        # --- Metrics Calculation ---
        # No need to check for NaN predictions as cross_val_predict handles folds internally
        labels_present = np.unique(y_true)
        accuracy = accuracy_score(y_true, y_pred)
        # Use weighted average; ensure labels argument covers all classes in y_true for consistency
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0, labels=labels_present
        )
        auc = np.nan # Default AUC
        # Calculate AUC only for binary cases with variance in both true and pred labels
        if len(labels_present) == 2 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            try: auc = roc_auc_score(y_true, y_pred)
            except ValueError: pass # Keep AUC as NaN if score is undefined

        subject_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
        print(f"  Metrics: Acc={subject_metrics['accuracy']:.3f}, Prc={subject_metrics['precision']:.3f}, Rec={subject_metrics['recall']:.3f}, F1={subject_metrics['f1']:.3f}")

        subject_metrics['subject_id'] = subject_id
        all_metrics_list.append(subject_metrics)
        # End of loop for subject_id

    # --- Aggregate, Plot, and Save Results (after subject loop) ---
    print("\n--- Aggregating and Saving Results ---")
    if not all_metrics_list:
        print("No subjects evaluated. Exiting.")
        return

    metrics_df = pd.DataFrame(all_metrics_list)
    metrics_melted = metrics_df.melt(id_vars='subject_id', var_name='metric_name', value_name='metric_value')
    metrics_melted.dropna(subset=['metric_value'], inplace=True)

    if metrics_melted.empty:
        print("No valid metrics to plot.")
    else:
        plt.figure(figsize=(15, 8))
        sns.barplot(data=metrics_melted, x='subject_id', y='metric_value', hue='metric_name')
        plt.title('Per-Subject KNN Performance (Landscape LOOCV)')
        plt.xlabel('Subject ID'); plt.ylabel('Metric Value')
        unique_subjects_count = metrics_df['subject_id'].nunique()
        if unique_subjects_count > 15: plt.xticks(rotation=90, fontsize=8)
        else: plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric'); plt.tight_layout()

        plot_filename = os.path.join(OUTPUT_DIR, "subject_knn_landscape_metrics.png")
        csv_filename = os.path.join(OUTPUT_DIR, "subject_knn_landscape_metrics.csv")
        plt.savefig(plot_filename)
        metrics_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved:")
        print(f"  - Metrics CSV: {csv_filename}")
        print(f"  - Plot PNG: {plot_filename}")

    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    all_metrics_list = []
    main()
