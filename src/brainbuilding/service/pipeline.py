from sklearn.base import ClassifierMixin
from brainbuilding.core.config import CHANNELS_TO_KEEP
from brainbuilding.core.transformers import AugmentedDataset, Covariances, EyeRemoval, ParallelTransportTransformer, SimpleWhiteningTransformer, StructuredArrayBuilder, StructuredToArray


import numpy as np
from pyriemann.spatialfilters import CSP
from sklearn.svm import SVC


from dataclasses import dataclass
from typing import Any, Dict, List, Type, Optional


@dataclass
class PipelineStep:
    name: str
    component_class: Type
    is_classifier: bool = False
    init_params: Optional[Dict[str, Any]] = None
    requires_fit: bool = True

    def wrap_component(self, component) -> "PipelineStepBase":
        """Wrap transformer or classifier with unified interface"""
        if self.is_classifier:
            return ClassifierWrapper(component)
        else:
            return TransformerWrapper(component)


@dataclass
class PipelineConfig:
    steps: List["PipelineStep"]

    def partial_fit_components(
        self, data: np.ndarray, fitted_components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single forward pass partial-fit and transform sequentially for available steps."""
        updated_components: Dict[str, Any] = {}
        current: np.ndarray = data

        for step in self.steps:
            component = fitted_components.get(step.name)
            if component is None:
                if not step.requires_fit:
                    init_kwargs = step.init_params or {}
                    component = step.component_class(**init_kwargs)
                    updated_components[step.name] = component
                else:
                    break
            else:
                updated_components[step.name] = component

            wrapped = step.wrap_component(component)
            if (
                step.requires_fit
                and hasattr(component, "partial_fit")
                and not step.is_classifier
            ):
                wrapped.partial_fit(current)
            current = wrapped.transform(current)

        return updated_components

    def fit_next_stateful(
        self,
        data: np.ndarray,
        fitted_components: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fit only the first stateful (requires_fit=True) step that isn't fitted yet.

        Stateless steps are instantiated as needed to prepare the input (single forward pass).
        No transforms are applied beyond the fitted step.
        """
        newly_fitted: Dict[str, Any] = {}
        current: np.ndarray = data

        for step in self.steps:
            component = (
                fitted_components.get(step.name)
                if step.name in fitted_components
                else newly_fitted.get(step.name)
            )

            if component is None:
                init_kwargs = step.init_params or {}
                component = step.component_class(**init_kwargs)
                if step.requires_fit and not step.is_classifier:
                    component.fit(current)
                    newly_fitted[step.name] = component
                    break
                else:
                    newly_fitted[step.name] = component

            wrapped = step.wrap_component(component)
            current = wrapped.transform(current)

        return newly_fitted

    @classmethod
    def hybrid_pipeline(cls):
        return cls(
            steps=[
                PipelineStep(
                    name="ica",
                    component_class=EyeRemoval,
                    init_params={
                        "n_components": None,
                        "remove_veog": True,
                        "remove_heog": True,
                        "random_state": 42,
                    },
                    requires_fit=True,
                ),
                PipelineStep(
                    name="whitening",
                    component_class=SimpleWhiteningTransformer,
                    init_params={"n_channels": len(CHANNELS_TO_KEEP)},
                    requires_fit=True,
                ),
                PipelineStep(
                    name="augmentation",
                    component_class=AugmentedDataset,
                    init_params={"order": 4, "lag": 8},
                    requires_fit=False,
                ),
                PipelineStep(
                    name="covariance",
                    component_class=Covariances,
                    is_classifier=False,
                    init_params={"estimator": "oas"},
                    requires_fit=False,
                ),
                PipelineStep(
                    name="to_structured",
                    component_class=StructuredArrayBuilder,
                    init_params={"subject_id": 0, "sample_field": "sample"},
                    requires_fit=False,
                ),
                PipelineStep(
                    name="pt",
                    component_class=ParallelTransportTransformer,
                    init_params={
                        "include_means": False,
                        "prevent_subject_drift": True,
                        "subject_min_samples_for_transform": 5,
                    },
                    requires_fit=True,
                ),
                PipelineStep(
                    name="from_structured",
                    component_class=StructuredToArray,
                    init_params={"field": "sample"},
                    requires_fit=False,
                ),
                PipelineStep(
                    name="csp",
                    component_class=CSP,
                    init_params={
                        "nfilter": 10,
                        "log": False,
                        "metric": "riemann",
                    },
                    requires_fit=True,
                ),
                PipelineStep(
                    name="classifier",
                    component_class=SVC,
                    is_classifier=True,
                    init_params={
                        "kernel": "rbf",
                        "C": 0.696,
                        "gamma": 0.035,
                        "class_weight": None,
                        "probability": True,
                    },
                    requires_fit=True,
                ),
            ]
        )


class PipelineStepBase:
    """Abstract base for pipeline steps to unify transformers and classifiers"""

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Unified interface for both transform and predict operations"""
        raise NotImplementedError

    def partial_fit(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None
    ):
        """Unified interface for online learning"""
        raise NotImplementedError


class TransformerWrapper(PipelineStepBase):
    def __init__(self, transformer):
        self.transformer = transformer

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.transformer.transform(data)

    def partial_fit(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None
    ):
        if hasattr(self.transformer, "partial_fit"):
            self.transformer.partial_fit(data)


class ClassifierWrapper(PipelineStepBase):
    def __init__(self, classifier: ClassifierMixin):
        self.classifier: ClassifierMixin = classifier

    def transform(self, data: np.ndarray) -> np.ndarray:
        # Return [predicted_label, predicted_probability] for downstream consumption
        try:
            if hasattr(self.classifier, "predict_proba"):
                proba = self.classifier.predict_proba(data)
                if proba.ndim == 2 and proba.shape[0] >= 1:
                    pred_idx = int(np.argmax(proba[0]))
                    pred = (
                        self.classifier.classes_[pred_idx]
                        if hasattr(self.classifier, "classes_")
                        else pred_idx
                    )
                    conf = float(np.max(proba[0]))
                    return np.array([pred, conf])
        except Exception:
            pass

        pred_arr = self.classifier.predict(data)
        pred = (
            int(pred_arr[0]) if hasattr(pred_arr, "__len__") else int(pred_arr)
        )
        return np.array([pred, 1.0])

    def partial_fit(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None
    ):
        if hasattr(self.classifier, "partial_fit") and labels is not None:
            self.classifier.partial_fit(data, labels)