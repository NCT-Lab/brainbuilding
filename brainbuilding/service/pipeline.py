# NOTE: Keep imports minimal; avoid unused project-wide constants here
from brainbuilding.core.transformers import (
    AugmentedDataset,
    Covariances,
    EyeRemoval,
    ParallelTransportTransformer,
    SimpleWhiteningTransformer,
    StructuredArrayBuilder,
    StructuredColumnTransformer,
    StructuredToArray,
    CustomSVC,
    Resampler,
)


import numpy as np
from pyriemann.spatialfilters import CSP  # type: ignore
from sklearn.svm import SVC  # type: ignore


from dataclasses import dataclass
from typing import Any, Dict, List, Type, Optional, Tuple
import yaml  # type: ignore
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
from typing import Any as _PydanticAny
import logging
import os
from joblib import load as joblib_load  # type: ignore

LOG_PIPELINE = logging.getLogger("brainbuilding.service.pipeline")


@dataclass
class PipelineStep:
    name: str
    component_class: Type
    init_params: Optional[Dict[str, Any]] = None
    requires_fit: bool = True
    calibration: bool = False
    apply_method: str = "transform"
    use_column: Optional[str] = None
    pass_class_label: bool = False
    is_fit_during_runtime: bool = False


@dataclass
class PipelineConfig:
    steps: List["PipelineStep"]

    def partial_fit_components(
        self, data: np.ndarray, fitted_components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single forward pass partial-fit and transform sequentially for
        available steps.
        """
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
                    # TODO: raise error may be?
                    LOG_PIPELINE.error(
                        "Component %s not fit; cannot partial-fit. Aborting.",
                        step.name,
                    )
                    break
            else:
                updated_components[step.name] = component

            if step.requires_fit and hasattr(component, "partial_fit"):
                args = (
                    [current[step.use_column]]
                    if step.use_column is not None
                    else [current]
                )
                if step.pass_class_label:
                    args.append(current["label"])
                component.partial_fit(*args)

            # Apply forward using column-aware wrapping where needed
            if step.use_column is None:
                apply_fn = getattr(component, step.apply_method)
                try:
                    current = apply_fn(current)
                except ValueError:
                    LOG_PIPELINE.error(f"Error applying {step.name} during partial-fit")
                    return updated_components
            else:
                if step.apply_method in ("transform", "fit_transform"):
                    wrapper = StructuredColumnTransformer(
                        column=step.use_column,
                        transformer=component,
                    )
                    apply_fn = getattr(wrapper, step.apply_method)
                    try:
                        current = apply_fn(current)
                    except:
                        LOG_PIPELINE.error(f"Error applying {step.name} to {current.shape}")
                        return updated_components
                else:
                    # TODO: raise error may be?
                    LOG_PIPELINE.error(
                        (
                            "Component requires column application but lacks "
                            "transform/fit_transform. Returning empty components"
                        )
                    )
                    return updated_components

        return updated_components

    def fit_next_stateful(
        self,
        data: np.ndarray,
        fitted_components: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fit only the first stateful (requires_fit=True) step that isn't
        fitted yet.

        Stateless steps are instantiated as needed to prepare the input
        (single forward pass). No transforms are applied beyond the fitted
        step.
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
                if step.requires_fit:
                    args = (
                        [current[step.use_column]]
                        if step.use_column is not None
                        else [current]
                    )
                    if step.pass_class_label:
                        args.append(current["label"])
                    component.fit(*args)
                    newly_fitted[step.name] = component
                    break
                else:
                    newly_fitted[step.name] = component

            # Stateless or already-fitted: propagate forward once
            if step.use_column is None:
                apply_fn = getattr(component, step.apply_method)
                current = apply_fn(current)
            else:
                if step.apply_method in ("transform", "fit_transform"):
                    wrapper = StructuredColumnTransformer(
                        column=step.use_column,
                        transformer=component,
                    )
                    apply_fn = getattr(wrapper, step.apply_method)
                    current = apply_fn(current)
                else:
                    # TODO: raise error may be?
                    LOG_PIPELINE.error(
                        (
                            "Component requires column application but lacks "
                            "transform/fit_transform. Returning empty components"
                        )
                    )
                    return {}

        return newly_fitted

    def predict(
        self,
        data: np.ndarray,
        fitted_components: Dict[str, Any],
    ) -> Optional[Any]:
        current: np.ndarray = data
        for step in self.steps:
            component = fitted_components.get(step.name, None)

            if component is None:
                if not step.requires_fit:
                    init_kwargs = step.init_params or {}
                    component = step.component_class(**init_kwargs)
                else:
                    LOG_PIPELINE.error(
                        "Component %s not fit; cannot predict. Aborting.",
                        step.name,
                    )
                    return None

            if step.use_column is None:
                apply_fn = getattr(component, step.apply_method)
                current = apply_fn(current)
            else:
                if step.apply_method in ("transform", "fit_transform"):
                    wrapper = StructuredColumnTransformer(
                        column=step.use_column,
                        transformer=component,
                    )
                    apply_fn = getattr(wrapper, step.apply_method)
                    current = apply_fn(current)
                else:
                    raise ValueError(
                        (
                            "Component requires column application but lacks "
                            "transform/fit_transform"
                        )
                    )

        return current


    # -----------------------------
    # Calibration helpers
    # -----------------------------

    def get_calibration_steps(self) -> List["PipelineStep"]:
        """Return steps that are marked as calibration steps in the YAML."""
        return [s for s in self.steps if s.calibration]

    def get_training_steps(self) -> List["PipelineStep"]:
        """Return steps that are NOT marked as calibration steps."""
        return [s for s in self.steps if not s.calibration]

    def apply_calibration(
        self,
        data: np.ndarray,
        fitted_components: Dict[str, Any],
    ) -> np.ndarray:
        """Apply only calibration steps to the provided data.

        All required calibration components must be already fitted. Stateless
        components (requires_fit=False) will be instantiated on demand and
        cached into fitted_components for reuse.
        """
        current: np.ndarray = data
        for step in self.get_calibration_steps():
            component = fitted_components.get(step.name)
            if component is None:
                if step.requires_fit:
                    raise ValueError(
                        f"Calibration component '{step.name}' is not fitted"
                    )
                init_kwargs = step.init_params or {}
                component = step.component_class(**init_kwargs)
                fitted_components[step.name] = component

            if step.use_column is None:
                apply_fn = getattr(component, step.apply_method)
                current = apply_fn(current)
            else:
                if step.apply_method in ("transform", "fit_transform"):
                    wrapper = StructuredColumnTransformer(
                        column=step.use_column,
                        transformer=component,
                    )
                    apply_fn = getattr(wrapper, step.apply_method)
                    current = apply_fn(current)
                else:
                    raise ValueError(
                        (
                            "Component requires column application but lacks "
                            "transform/fit_transform"
                        )
                    )

        return current

    def training_only_config(self) -> "PipelineConfig":
        """Create a new PipelineConfig with non-calibration steps only."""
        return PipelineConfig(steps=self.get_training_steps())

    def fit_training(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit all non-calibration steps sequentially and return components.

        Stateless steps are instantiated and applied to propagate the data.
        """
        fitted: Dict[str, Any] = {}
        current: np.ndarray = data

        for step in self.get_training_steps():
            LOG_PIPELINE.info(f"Fitting step {step.name}")
            try:
                LOG_PIPELINE.info("%s", f"{current['sample'].shape=}")
            except Exception:
                LOG_PIPELINE.info("%s", f"{current.shape=}")
            init_kwargs = step.init_params or {}
            component = step.component_class(**init_kwargs)

            if step.requires_fit and hasattr(component, "fit"):
                args = (
                    [current[step.use_column]]
                    if step.use_column is not None
                    else [current]
                )
                if step.pass_class_label:
                    if labels is None:
                        raise ValueError(
                            (
                                "Labels are required for step '%s' but were not "
                                "provided"
                            ) % step.name
                        )
                    args.append(labels)
                component.fit(*args)
                fitted[step.name] = component

            # Propagate forward if apply method exists (column-aware)
            if hasattr(component, step.apply_method):
                if step.use_column is None:
                    apply_fn = getattr(component, step.apply_method)
                    current = apply_fn(current)
                else:
                    if step.apply_method in ("transform", "fit_transform"):
                        wrapper = StructuredColumnTransformer(
                            column=step.use_column,
                            transformer=component,
                        )
                        apply_fn = getattr(wrapper, step.apply_method)
                        current = apply_fn(current)
                    else:
                        raise ValueError(
                            (
                                "Component requires column application but lacks "
                                "transform/fit_transform"
                            )
                        )

        return fitted

    def apply_training(self, data: np.ndarray, fitted_components: Dict[str, Any]) -> np.ndarray:
        current: np.ndarray = data
        for step in self.steps:
            component = fitted_components.get(step.name, None)

            if component is None and step.requires_fit:
                raise ValueError(f"Component {step.name} is None")
            elif component is None:
                init_kwargs = step.init_params or {}
                component = step.component_class(**init_kwargs)
            if step.use_column is None:
                apply_fn = getattr(component, step.apply_method)
                current = apply_fn(current)
            else:
                if step.apply_method in ("transform", "fit_transform"):
                    wrapper = StructuredColumnTransformer(
                        column=step.use_column,
                        transformer=component,
                    )
                    apply_fn = getattr(wrapper, step.apply_method)
                    current = apply_fn(current)
                else:
                    raise ValueError(
                        (
                            "Component requires column application but lacks "
                            "transform/fit_transform"
                        )
                    )

        return current
        

# -----------------------------
# YAML-driven pipeline loading
# -----------------------------

class PipelineStepConfigModel(BaseModel):
    name: str
    component: str
    requires_fit: bool = True
    calibration: bool = False
    apply_method: str = "transform"
    init_params: Dict[str, Any] = Field(default_factory=dict)
    use_column: Optional[str] = None
    pass_class_label: bool = False
    is_fit_during_runtime: bool = False

    @field_validator("component")
    @classmethod
    def _validate_component(cls, v: str) -> str:
        if v not in COMPONENT_REGISTRY:
            raise ValueError(
                f"Unknown component class in pipeline config: {v}"
            )
        return v

    @field_validator("apply_method")
    @classmethod
    def _validate_apply_method(
        cls, v: str, info: _PydanticAny
    ) -> str:
        comp = getattr(info, "data", {}).get("component")
        if isinstance(comp, str) and comp in COMPONENT_REGISTRY:
            component_cls = COMPONENT_REGISTRY[comp]
            if not hasattr(component_cls, v):
                raise ValueError(
                    f"apply_method '{v}' is not available on component '{comp}'"
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
        "Resampler": Resampler,
        # Feature extraction / models
        "CSP": CSP,
        "SVC": SVC,
        "CustomSVC": CustomSVC,
    }


def load_pipeline_from_yaml(
    path: str,
    preload_dir: Optional[str] = None,
    enable_preload: bool = True,
) -> Tuple[PipelineConfig, Dict[str, Any]]:
    """Load a PipelineConfig and initial fitted components from a YAML file.

    Returns (pipeline_config, pretrained_components).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            "Invalid pipeline YAML: expected a mapping at top level"
        )
    cfg = PipelineYAMLConfigModel(**raw)

    steps: List[PipelineStep] = []
    pretrained: Dict[str, Any] = {}

    for s in cfg.steps:
        component_cls = COMPONENT_REGISTRY[s.component]
        step = PipelineStep(
            name=s.name,
            component_class=component_cls,
            init_params=s.init_params or {},
            requires_fit=s.requires_fit,
            calibration=s.calibration,
            apply_method=s.apply_method or "transform",
            use_column=s.use_column,
            pass_class_label=s.pass_class_label,
        )
        steps.append(step)

        # Optional preload of pretrained components from a single directory
        if enable_preload and preload_dir:
            candidate_path = os.path.join(preload_dir, f"{s.name}.joblib")
            if os.path.exists(candidate_path):
                loaded = joblib_load(candidate_path)
                if not isinstance(loaded, component_cls):
                    raise ValueError(
                        (
                            f"Pretrained artifact at '{candidate_path}' for step "
                            f"'{s.name}' has type {type(loaded).__name__}; "
                            f"expected {component_cls.__name__}"
                        )
                    )
                pretrained[s.name] = loaded

    return PipelineConfig(steps=steps), pretrained


class PipelineStepBase:  # legacy placeholder (removed wrappers)
    pass
