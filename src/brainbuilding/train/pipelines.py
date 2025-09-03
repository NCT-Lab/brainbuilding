from typing import Optional

import numpy as np
from sklearn.svm import SVC  # type: ignore
from pyriemann.spatialfilters import CSP  # type: ignore
from sklearn.decomposition import KernelPCA  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.pipeline import Pipeline


from ..core.transformers import (
    Covariances,
    ParallelTransportTransformer,
    StructuredCSP,
    SubjectWhiteningTransformer,
    AugmentedDataset,
    BackgroundFilterTransformer,
    ColumnSelector,
    TangentSpaceProjector,
    StructuredColumnTransformer,
)


N_NEIGHBORS = 25
N_FILTERS = 10


SIMPLE_CSP = [
    ("selector", ColumnSelector(fields="sample")),
    ("csp", CSP(nfilter=N_FILTERS, log=True, metric="riemann")),
]

PT_CSP_TANGENT = [
    ("pt", ParallelTransportTransformer(include_means=True)),
    (
        "csp",
        StructuredCSP(
            field="sample", nfilter=N_FILTERS, log=False, metric="riemann"
        ),
    ),
    ("tangent-space", TangentSpaceProjector()),
    ("selector", ColumnSelector(fields="sample")),
]

PT_CSP_LOG = [
    ("pt", ParallelTransportTransformer()),
    ("selector", ColumnSelector(fields="sample")),
    ("csp", CSP(nfilter=N_FILTERS, log=True, metric="riemann")),
]

CSP_PT_TANGENT = [
    (
        "csp",
        StructuredCSP(
            field="sample", nfilter=N_FILTERS, log=False, metric="riemann"
        ),
    ),
    ("pt", ParallelTransportTransformer(include_means=True)),
    ("tangent-space", TangentSpaceProjector()),
    ("selector", ColumnSelector(fields="sample")),
]

KPCA_REDUCTION = [
    ("kpca", KernelPCA(n_components=N_FILTERS, kernel="rbf", gamma=0.035)),
]

SVC_CLASSIFICATION = [
    (
        "svc",
        SVC(
            kernel="rbf",
            C=0.696,
            gamma=0.035,
            class_weight=None,
            probability=True,
        ),
    )
]

KNN_NO_DISTANCE_CLASSIFICATION = [
    (
        "knn-no-distance",
        KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="uniform"),
    )
]

KNN_DISTANCE_CLASSIFICATION = [
    (
        "knn-distance",
        KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance"),
    )
]


OAS_COV = [
    (
        "covariances",
        StructuredColumnTransformer(
            column="sample", transformer=Covariances(estimator="oas")
        ),
    ),
]

AUGMENTED_OAS_COV = [
    (
        "augmented_dataset",
        StructuredColumnTransformer(
            column="sample", transformer=AugmentedDataset(order=4, lag=8)
        ),
    ),
    (
        "covariances",
        StructuredColumnTransformer(
            column="sample", transformer=Covariances(estimator="oas")
        ),
    ),
]

WHITENING = [
    ("whitening", SubjectWhiteningTransformer()),
]

BACKGROUND_FILTER = [
    ("background_filter", BackgroundFilterTransformer()),
]

GENERAL_PIPELINE_STEPS = [
    [WHITENING],
    [BACKGROUND_FILTER],
    [AUGMENTED_OAS_COV],
    [SIMPLE_CSP],
    [[]],
    [SVC_CLASSIFICATION],
]


class OnlinePipeline(Pipeline):
    """
    A pipeline that supports partial_fit for online learning.

    This class extends scikit-learn's Pipeline to support partial_fit,
    which is useful for online learning scenarios. It will call partial_fit
    on all steps that implement it, while maintaining the regular fit behavior
    for steps that don't support partial_fit.

    Parameters
    ----------
    steps : list of (str, estimator) tuples
        List of (name, estimator) tuples that are chained in sequence.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.
    """

    def partial_fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params
    ) -> "OnlinePipeline":
        """
        Fit the pipeline on a batch of data.

        This method will call partial_fit on all steps that implement it,
        while maintaining the regular fit behavior for steps that don't support
        partial_fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.
        **fit_params : dict of string -> object
            Parameters passed to the partial_fit method of each step.

        Returns
        -------
        self : OnlinePipeline
            This estimator.
        """
        Xt = X
        for name, step in self.steps:
            if hasattr(step, "partial_fit"):
                # If step supports partial_fit, use it
                step.partial_fit(Xt, y, **fit_params)

            # Transform the data for the next step
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)

        return self
