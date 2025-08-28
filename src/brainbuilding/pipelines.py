from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import numpy as np
from pyriemann.classification import KNearestNeighbor
from pyriemann.spatialfilters import CSP
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from .eye_removal import EyeRemoval
from .transformers import (
    AugmentedDataset,
    BackgroundFilterTransformer,
    ColumnSelector,
    Covariances,
    ParallelTransportTransformer,
    ReferenceKNN,
    StructuredColumnTransformer,
    StructuredCSP,
    SubjectWhiteningTransformer,
    TangentSpaceProjector,
    reference_weighted_euclidean_distance,
)


@dataclass
class PipelineConfig:
    steps: List['PipelineStep']

    def fit_components(
        self, data: np.ndarray, fitted_components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Progressive fitting of pipeline components with proper dependency handling"""
        newly_fitted = {}
        processed_data = data.copy()

        for step in self.steps:
            if step.name in fitted_components:
                component = fitted_components[step.name]
                wrapped = step.wrap_component(component)
                processed_data = wrapped.transform(processed_data)

        for step in self.steps:
            if step.name not in fitted_components:
                component = step.component_class()
                component.fit(processed_data)
                newly_fitted[step.name] = component
                wrapped = step.wrap_component(component)
                processed_data = wrapped.transform(processed_data)

        return newly_fitted

    def partial_fit_components(
        self, data: np.ndarray, fitted_components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Progressive partial fitting with proper dependency handling"""
        updated_components = {}
        processed_data = data.copy()

        for step in self.steps:
            component = fitted_components[step.name]
            wrapped_component = step.wrap_component(component)

            if hasattr(component, "partial_fit") and not step.is_classifier:
                wrapped_component.partial_fit(processed_data)

            processed_data = wrapped_component.transform(processed_data)
            updated_components[step.name] = component

        return updated_components

    @classmethod
    def hybrid_pipeline(cls):
        return cls(
            steps=[
                PipelineStep(name="ica", component_class=EyeRemoval),
                PipelineStep(
                    name="whitening", component_class=SubjectWhiteningTransformer
                ),
                PipelineStep(
                    name="covariance", component_class=Covariances, is_classifier=False
                ),
                PipelineStep(
                    name="pt", component_class=ParallelTransportTransformer
                ),
                PipelineStep(name="csp", component_class=StructuredCSP),
                PipelineStep(
                    name="classifier", component_class=SVC, is_classifier=True
                ),
            ]
        )


class PipelineStepBase:
    """Abstract base for pipeline steps to unify transformers and classifiers"""

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Unified interface for both transform and predict operations"""
        raise NotImplementedError

    def partial_fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        """Unified interface for online learning"""
        raise NotImplementedError


class TransformerWrapper(PipelineStepBase):
    def __init__(self, transformer):
        self.transformer = transformer

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.transformer.transform(data)

    def partial_fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        if hasattr(self.transformer, "partial_fit"):
            self.transformer.partial_fit(data)


class ClassifierWrapper(PipelineStepBase):
    def __init__(self, classifier):
        self.classifier = classifier

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.classifier.predict(data)

    def partial_fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        if hasattr(self.classifier, "partial_fit") and labels is not None:
            self.classifier.partial_fit(data, labels)


@dataclass
class PipelineStep:
    name: str
    component_class: Type
    is_classifier: bool = False

    def wrap_component(self, component) -> PipelineStepBase:
        """Wrap transformer or classifier with unified interface"""
        if self.is_classifier:
            return ClassifierWrapper(component)
        else:
            return TransformerWrapper(component)


N_NEIGHBORS = 25
N_FILTERS = 10

# Parallel Transport -> Tangent Space -> KNN
# PT_TANGENT_KNN_STEPS = [
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('project-tangent', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # Parallel Transport -> Tangent Space -> KernelPCA -> KNN
# PT_TANGENT_KPCA_KNN_STEPS = [
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('project-tangent', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('kpca', KernelPCA(n_components=N_FILTERS, kernel='rbf', gamma=0.035)),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # Parallel Transport -> CSP (no log) -> Riemannian KNN
# PT_CSP_KNN_STEPS = [
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
#     ('tangent-space', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # Parallel Transport -> CSP (with log) -> SVC
# PT_CSP_SVC_STEPS = [
#     ('pt', ParallelTransportTransformer()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
#     ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
# ]

# # CSP -> Parallel Transport -> Tangent Space -> SVC
# CSP_PT_TANGENT_SVC_STEPS = [
#     ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('tangent-space', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
# ]

# # CSP -> Parallel Transport -> Tangent Space -> KNN
# CSP_PT_TANGENT_KNN_STEPS = [
#     ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('tangent-space', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # CSP -> SVC
# CSP_SVC_STEPS = [
#     ('selector', ColumnSelector(fields='sample')),
#     ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
#     ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
# ]


SIMPLE_CSP = [
    ('selector', ColumnSelector(fields='sample')),
    ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
]

PT_CSP_TANGENT = [
    ('pt', ParallelTransportTransformer(include_means=True)),
    ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
    ('tangent-space', TangentSpaceProjector()),
    ('selector', ColumnSelector(fields='sample')),
]

PT_CSP_LOG = [
    ('pt', ParallelTransportTransformer()),
    ('selector', ColumnSelector(fields='sample')),
    ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
]

CSP_PT_TANGENT = [
    ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
    ('pt', ParallelTransportTransformer(include_means=True)),
    ('tangent-space', TangentSpaceProjector()),
    ('selector', ColumnSelector(fields='sample')),
]

KPCA_REDUCTION = [
    ('kpca', KernelPCA(n_components=N_FILTERS, kernel='rbf', gamma=0.035)),
]

SVC_CLASSIFICATION = [
    ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
]

KNN_NO_DISTANCE_CLASSIFICATION = [
    ('knn-no-distance', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='uniform'))
]

KNN_DISTANCE_CLASSIFICATION = [
    ('knn-distance', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
]


OAS_COV = [
    ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
]

AUGMENTED_OAS_COV = [
    ('augmented_dataset', StructuredColumnTransformer(column='sample', transformer=AugmentedDataset(order=4, lag=8))),
    ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
]

WHITENING = [
    ('whitening', SubjectWhiteningTransformer()),
]

BACKGROUND_FILTER = [
    ('background_filter', BackgroundFilterTransformer()),
]

GENERAL_PIPELINE_STEPS = [
    [
        WHITENING
    ],
    [
        BACKGROUND_FILTER
    ],
    [
        AUGMENTED_OAS_COV
    ],
    [
        SIMPLE_CSP
    ],
    [
        []
    ],
    [
        SVC_CLASSIFICATION
    ]
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
    
    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> 'OnlinePipeline':
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
            if hasattr(step, 'partial_fit'):
                # If step supports partial_fit, use it
                step.partial_fit(Xt, y, **fit_params)

            # Transform the data for the next step
            if hasattr(step, 'transform'):
                Xt = step.transform(Xt)
                
        return self