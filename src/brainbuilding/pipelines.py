from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import numpy as np
from sklearn.svm import SVC  # type: ignore
from pyriemann.spatialfilters import CSP  # type: ignore
from sklearn.decomposition import KernelPCA  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from .eye_removal import EyeRemoval
from .transformers import (
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


@dataclass
class PipelineConfig:
    steps: List['PipelineStep']

    def fit_components(
        self, data: np.ndarray, fitted_components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Progressive fitting of pipeline components with proper dependency handling"""
        newly_fitted: Dict[str, Any] = {}

        # Walk steps in order; for each step i, transform input through
        # only previously available steps (existing or newly fitted in this call),
        # then fit step i if it's missing.
        for i, step in enumerate(self.steps):
            # Build processed_data by passing through prior steps 0..i-1
            processed_data: Optional[np.ndarray] = data
            for j in range(i):
                prev = self.steps[j]
                prev_component = (
                    fitted_components.get(prev.name)
                    if prev.name in fitted_components
                    else newly_fitted.get(prev.name)
                )
                if prev_component is None:
                    # Prior step not yet available; cannot fit step i in this pass
                    processed_data = None
                    break
                if processed_data is not None:
                    processed_data = prev.wrap_component(prev_component).transform(
                        processed_data
                    )

            if processed_data is None:
                continue

            if processed_data is not None and step.name not in fitted_components:
                component = step.component_class()
                component.fit(processed_data)
                newly_fitted[step.name] = component

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