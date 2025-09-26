import numpy as np
from pyriemann.classification import KNearestNeighbor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
from typing import List, Any, Tuple
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from ..core.transformers import (
    ParallelTransportTransformer,
    TangentSpaceProjector,
)
from .pipelines import OnlinePipeline


def evaluate_subject_with_riemannian_knn(X, y, event_ids, k=5, verbose=False):
    """
    Evaluate a subject's data using Riemannian KNN classification.

    Args:
        X: array-like of shape (n_samples, n_channels, n_channels)
            Covariance matrices for each sample
        y: array-like of shape (n_samples,)
            Labels for each sample
        event_ids: array-like of shape (n_samples,)
            Event IDs for each sample
        k: int, default=5
            Number of neighbors to use for KNN

    Returns:
        tuple: (metrics, y_true, y_pred)
            metrics: dict
                Dictionary containing evaluation metrics
            y_true: list
                List of true labels
            y_pred: list
                List of predicted labels
    """

    n_samples = len(X)
    y_true = []
    y_pred = []

    # Perform leave-one-out cross-validation
    for i in tqdm(
        range(n_samples), desc="Evaluating samples", disable=not verbose
    ):
        # Initialize the Riemannian KNN classifier
        clf = KNearestNeighbor(n_neighbors=k, metric="riemann")

        # Get the test sample's event ID
        test_event_id = event_ids[i]

        # Create mask to exclude samples from the same event
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False  # Exclude the test sample itself

        # Exclude samples from the same event
        for j in range(n_samples):
            if event_ids[j] == test_event_id and j != i:
                train_mask[j] = False

        # If no valid training samples after filtering, use all except self
        if not np.any(train_mask):
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[i] = False

        # Get training and test data
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[i : i + 1]  # Keep as 3D array

        # Fit and predict
        clf.fit(X_train, y_train)
        pred_label = clf.predict(X_test)[0]

        # Store results
        y_true.append(y[i])
        y_pred.append(pred_label)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    return metrics, y_true, y_pred


def evaluate_pipeline_with_adaptation(
    X: np.ndarray,
    y: np.ndarray,
    covariance_transformer: Any,
    pipeline_steps: List[Tuple[str, Any]],
    use_background: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a pipeline with adaptation using subject-wise cross-validation.

    Parameters
    ----------
    X : structured array
        Input data with fields:
        - sample: (n_channels, n_times) array
        - subject_id: int
        - event_id: int
        - is_background: int (0 or 1)
    y : array-like
        Target labels
    covariance_transformer : object
        Transformer to convert samples to covariance matrices
    pipeline_steps : list of tuples
        Steps for the pipeline
    use_background : bool, default=False
        Whether to use background samples for adaptation

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - First DataFrame contains per-subject metrics
        - Second DataFrame contains descriptive statistics of the metrics
    """
    # whitening_transformer = SubjectWhiteningTransformer()
    # X = whitening_transformer.fit_transform(X)
    # Create pipeline
    pipeline = Pipeline(pipeline_steps)

    # Find ParallelTransportTransformer in the pipeline
    pt_transformer = None
    for name, step in pipeline_steps:
        if isinstance(
            step, (ParallelTransportTransformer, TangentSpaceProjector)
        ):
            pt_transformer = step
            break

    if pt_transformer is None:
        raise ValueError(
            "Pipeline must contain a ParallelTransportTransformer"
        )

    covariance_matrix_shape = covariance_transformer.fit_transform(
        X["sample"][0][np.newaxis, :, :]
    )[0].shape
    # Convert all samples to covariance matrices upfront
    X_cov = np.zeros(
        len(X),
        dtype=[
            ("sample", np.float64, covariance_matrix_shape),
            ("subject_id", np.int64),
            ("event_id", np.int64),
            ("is_background", np.int64),
        ],
    )
    X_cov["subject_id"] = X["subject_id"]
    X_cov["event_id"] = X["event_id"]
    X_cov["is_background"] = X["is_background"]

    # Convert all samples to covariance matrices
    X_cov["sample"] = covariance_transformer.fit_transform(X["sample"])
    mean_estimator = ParallelTransportTransformer()
    mean_estimator.fit(X_cov[X_cov["is_background"] == 0])

    subject_means = mean_estimator.get_subject_means()
    subject_counts = mean_estimator.get_subject_counts()

    # Prepare cross-validation
    subject_ids = X_cov["subject_id"]
    unique_subjects = np.unique(subject_ids)
    cv = GroupKFold(n_splits=len(unique_subjects))

    # Store results
    results = []

    # Perform cross-validation
    for train_idx, test_idx in tqdm(
        cv.split(X_cov, y, subject_ids),
        total=len(unique_subjects),
        desc="Cross-validation",
    ):
        # Create pipeline
        pipeline = Pipeline(pipeline_steps)

        # Find ParallelTransportTransformer in the pipeline
        pt_transformer = None
        for name, step in pipeline.steps:
            if isinstance(
                step, (ParallelTransportTransformer, TangentSpaceProjector)
            ):
                pt_transformer = step
                break

        # Get train and test data
        X_train = X_cov[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Triple the amount of samples by splitting each sample into 3 segments using numpy operations
        n_samples = len(X_test)

        # Create expanded structured array
        X_test_expanded = np.zeros(n_samples * 3, dtype=X_test.dtype)

        # Copy metadata for all samples (will be repeated for each segment)
        for field in ["subject_id", "event_id", "is_background"]:
            X_test_expanded[field] = np.repeat(X_test[field], 3)
        y_test = np.repeat(y_test, 3)

        # Split the EEG data into 3 segments for each sample
        original_samples = X_test["sample"]
        segments = np.array(
            [original_samples[:, :, i * 250 : (i + 1) * 250] for i in range(3)]
        )

        # Reshape to get the desired order: (n_samples*3, 18, 250)
        X_test_reduced = segments.transpose(1, 0, 2, 3).reshape(
            n_samples * 3, 18, 250
        )

        # Convert all samples to covariance matrices upfront
        X_test_cov = np.zeros(
            len(X_test_reduced),
            dtype=[
                ("sample", np.float64, covariance_matrix_shape),
                ("subject_id", np.int64),
                ("event_id", np.int64),
                ("is_background", np.int64),
            ],
        )
        X_test_cov["subject_id"] = X_test_expanded["subject_id"]
        X_test_cov["event_id"] = X_test_expanded["event_id"]
        X_test_cov["is_background"] = X_test_expanded["is_background"]

        # Convert all samples to covariance matrices
        X_test_cov["sample"] = covariance_transformer.fit_transform(
            X_test_reduced
        )
        X_test = X_test_cov

        # Get subject ID for this fold
        assert len(np.unique(X_test["subject_id"])) == 1
        subject_id = np.unique(X_test["subject_id"])[0]

        # Update PT transformer with current subject's mean and count (for optimization)
        temp_subject_means = subject_means.copy()
        temp_subject_counts = subject_counts.copy()
        temp_subject_means.pop(subject_id, None)
        temp_subject_counts.pop(subject_id, None)
        pt_transformer.set_subject_means_and_counts(
            temp_subject_means, temp_subject_counts
        )

        # Fit pipeline on non-background training data
        train_mask = X_train["is_background"] == 0
        pipeline.fit(X_train[train_mask], y_train[train_mask])

        # Get background samples for adaptation if requested
        if use_background:
            background_mask = X_test["is_background"] == 1
            if np.any(background_mask):
                # Update PT transformer with background samples
                pt_transformer.partial_fit(X_test[background_mask])

        # Predict on non-background test samples one by one
        test_mask = X_test["is_background"] == 0
        y_pred = []
        y_true = []
        y_score = []

        for i in np.where(test_mask)[0]:
            # Update PT transformer with current sample
            pt_transformer.partial_fit(
                X_test[i : i + 1], custom_weights=np.array([1 / 3])
            )

            # Predict
            pred = pipeline.predict(X_test[i : i + 1])[0]

            try:
                score = pipeline.predict_proba(X_test[i : i + 1])[0]
                score = np.max(score, axis=1)
            except Exception:
                score = np.nan

            y_pred.append(pred)
            y_true.append(y_test[i])
            y_score.append(score)

        # Calculate metrics
        metrics = {
            "subject": subject_id,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if not np.isnan(y_score).any():
            metrics["auc"] = roc_auc_score(y_true, y_score)
        else:
            metrics["auc"] = np.nan

        results.append(metrics)

    # Create per-subject metrics DataFrame
    metrics_df = pd.DataFrame(results)
    # Reshape to have subject, measure, value columns
    metrics_df = metrics_df.melt(
        id_vars=["subject"], var_name="measure", value_name="value"
    )

    # Calculate descriptive statistics
    desc_stats = (
        metrics_df.groupby("measure")["value"].describe().reset_index()
    )
    print(desc_stats)

    return metrics_df, desc_stats


def evaluate_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_steps: List[Tuple[str, Any]],
    return_predictions: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a pipeline with adaptation using subject-wise cross-validation.

    Parameters
    ----------
    X : structured array
        Input data with fields:
        - sample: (n_channels, n_times) array
        - subject_id: int
        - event_id: int
        - is_background: int (0 or 1)
    y : array-like
        Target labels
    pipeline_steps : list of tuples
        Steps for the pipeline
    return_predictions : bool, default=False
        If True, return detailed predictions instead of aggregated metrics

    Returns
    -------
    pd.DataFrame
        If return_predictions=False: DataFrame with per-subject metrics reshaped to long format
        If return_predictions=True: DataFrame with columns ['true_label', 'predicted_label', 'subject_id', 'sample_index']
    """
    # Create pipeline
    pipeline = OnlinePipeline(pipeline_steps)

    # Prepare cross-validation
    subject_ids = X["subject_id"]
    unique_subjects = np.unique(subject_ids)
    cv = GroupKFold(n_splits=len(unique_subjects))

    # Store results
    results = []
    predictions = []

    # Perform cross-validation
    for train_idx, test_idx in tqdm(
        cv.split(X, y, subject_ids),
        total=len(unique_subjects),
        desc="Cross-validation",
    ):
        train_mask = np.zeros(len(X), dtype=bool) | (X["is_background"] == 1)
        train_mask[train_idx] = True
        train_mask_labels = np.zeros(len(X), dtype=bool)
        train_mask_labels[train_idx] = True
        train_mask_labels = train_mask_labels & (X["is_background"] == 0)

        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[test_idx] = True
        test_mask[X["is_background"] == 1] = False
        test_mask_labels = np.zeros(len(X), dtype=bool)
        test_mask_labels[test_idx] = True
        test_mask_labels = test_mask_labels & (X["is_background"] == 0)

        # Get train and test data
        X_train = X[train_mask]
        y_train = y[train_mask_labels]
        X_test = X[test_mask]
        y_test = y[test_mask_labels]

        # Get subject ID for this fold
        assert len(np.unique(X_test["subject_id"])) == 1
        subject_id = np.unique(X_test["subject_id"])[0]

        pipeline.fit(X_train, y_train)

        y_pred = []
        y_true = []
        y_score = []

        for i in range(len(X_test)):
            # Update PT transformer with current sample
            pipeline.partial_fit(X_test[i : i + 1])

            # Predict
            pred = pipeline.predict(X_test[i : i + 1])[0]

            try:
                score = pipeline.predict_proba(X_test[i : i + 1])[0]
                score = np.nanmax(score)
            except Exception:
                score = np.nan

            y_pred.append(pred)
            y_true.append(y_test[i])
            y_score.append(score)

            # Store detailed predictions if requested
            if return_predictions:
                predictions.append(
                    {
                        "true_label": y_test[i],
                        "predicted_label": pred,
                        "subject_id": subject_id,
                        "sample_index": i,
                    }
                )

        # Calculate metrics only if not returning predictions
        if not return_predictions:
            metrics = {
                "subject": subject_id,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
            if not np.isnan(y_score).any():
                metrics["auc"] = roc_auc_score(y_true, y_score)
            else:
                metrics["auc"] = np.nan

            results.append(metrics)

    if return_predictions:
        return pd.DataFrame(predictions)
    else:
        # Create per-subject metrics DataFrame
        metrics_df = pd.DataFrame(results)
        # Reshape to have subject, measure, value columns
        metrics_df = metrics_df.melt(
            id_vars=["subject"], var_name="measure", value_name="value"
        )

        return metrics_df
