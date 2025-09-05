from sklearn.base import ClassifierMixin
from brainbuilding.core.config import CHANNELS_TO_KEEP
from brainbuilding.core.transformers import (
    AugmentedDataset,
    Covariances,
    EyeRemoval,
    ParallelTransportTransformer,
    SimpleWhiteningTransformer,
    StructuredArrayBuilder,
    StructuredToArray,
)


import numpy as np
from pyriemann.spatialfilters import CSP
from sklearn.svm import SVC


from dataclasses import dataclass
from typing import Any, Dict, List, Type, Optional, Tuple
import yaml  # type: ignore
from pydantic import BaseModel, Field, field_validator  # type: ignore


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

# -----------------------------
# YAML-driven pipeline loading
# -----------------------------

class PipelineStepConfigModel(BaseModel):
    name: str
    component: str
    # TODO: remove this field and its usage entirely
    is_classifier: bool = False
    requires_fit: bool = True
    calibration: bool = False
    init_params: Dict[str, Any] = Field(default_factory=dict)
    pretrained_path: Optional[str] = None

    @field_validator("component")
    @classmethod
    def _validate_component(cls, v: str) -> str:
        if v not in COMPONENT_REGISTRY:
            raise ValueError(
                f"Unknown component class in pipeline config: {v}"
            )
        return v


class PipelineYAMLConfigModel(BaseModel):
    steps: List[PipelineStepConfigModel] = Field(default_factory=list)


# Single source of truth registry constant for static tools and validators
COMPONENT_REGISTRY: Dict[str, Type] = {
        # Preprocessing and transformation
        "EyeRemoval": EyeRemoval,
        "SimpleWhiteningTransformer": SimpleWhiteningTransformer,
        "AugmentedDataset": AugmentedDataset,
        "Covariances": Covariances,
        "StructuredArrayBuilder": StructuredArrayBuilder,
        "ParallelTransportTransformer": ParallelTransportTransformer,
        "StructuredToArray": StructuredToArray,
        # Feature extraction / models
        "CSP": CSP,
        "SVC": SVC,
    }


def load_pipeline_from_yaml(path: str) -> Tuple[PipelineConfig, Dict[str, Any]]:
    """Load a PipelineConfig and initial fitted components from a YAML file.

    Returns (pipeline_config, pretrained_components).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Invalid pipeline YAML: expected a mapping at top level")
    cfg = PipelineYAMLConfigModel(**raw)

    steps: List[PipelineStep] = []
    pretrained: Dict[str, Any] = {}

    for s in cfg.steps:
        component_cls = COMPONENT_REGISTRY[s.component]
        step = PipelineStep(
            name=s.name,
            component_class=component_cls,
            is_classifier=s.is_classifier,
            init_params=s.init_params or {},
            requires_fit=s.requires_fit,
        )
        steps.append(step)

    return PipelineConfig(steps=steps), pretrained


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