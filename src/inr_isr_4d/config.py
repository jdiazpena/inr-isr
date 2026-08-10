"""Strict configuration contract for the additive 4D execution path."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


def _only(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown fields in {section}: {unknown}")


def _finite_positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int = 256
    hidden_layers: int = 3
    activation: str = "sine"
    first_omega_0: float = 5.0
    hidden_omega_0: float = 5.0

    def validate(self) -> None:
        if self.hidden_features <= 0 or self.hidden_layers < 0:
            raise ValueError("Model widths must be positive and hidden_layers non-negative.")
        if self.activation not in {"sine", "tanh", "relu", "softplus"}:
            raise ValueError(f"Unsupported activation: {self.activation}")
        _finite_positive(self.first_omega_0, "first_omega_0")
        _finite_positive(self.hidden_omega_0, "hidden_omega_0")


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 1.0e-4
    num_steps: int = 10_000
    data_batch_size: int = 4096
    seed: int = 0

    def validate(self) -> None:
        _finite_positive(self.learning_rate, "learning_rate")
        if self.num_steps <= 0 or self.data_batch_size <= 0 or self.seed < 0:
            raise ValueError("num_steps/data_batch_size must be positive and seed non-negative.")


@dataclass(frozen=True)
class CollocationConfig:
    mode: str = "sobol"
    pool_size: int = 65_536
    batch_size: int = 8192
    derivative_microbatch_size: int = 2048
    resample_every: int = 0
    seed: int = 0
    support_oversample_factor: int = 4
    domain_lower: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    domain_upper: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain_lower", tuple(float(value) for value in self.domain_lower))
        object.__setattr__(self, "domain_upper", tuple(float(value) for value in self.domain_upper))

    def validate(self) -> None:
        if self.mode not in {"data_coordinates", "sobol", "support_aware"}:
            raise ValueError(f"Unsupported collocation mode: {self.mode}")
        for name in ("pool_size", "batch_size", "derivative_microbatch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"collocation.{name} must be positive.")
        if self.batch_size > self.pool_size:
            raise ValueError("collocation.batch_size cannot exceed pool_size.")
        if self.derivative_microbatch_size > self.batch_size:
            raise ValueError("derivative_microbatch_size cannot exceed collocation batch_size.")
        if self.resample_every < 0 or self.seed < 0 or self.support_oversample_factor < 1:
            raise ValueError("Collocation interval/seed/factor values are invalid.")
        if len(self.domain_lower) != 4 or len(self.domain_upper) != 4:
            raise ValueError("Collocation domain bounds must each contain four values in x, y, z, t order.")
        for lower, upper in zip(self.domain_lower, self.domain_upper):
            if not math.isfinite(lower) or not math.isfinite(upper) or not -1.0 <= lower < upper <= 1.0:
                raise ValueError("Each normalized collocation bound must satisfy -1 <= lower < upper <= 1.")


@dataclass(frozen=True)
class DerivativePriorConfig:
    mode: str = "none"
    weight: float = 0.0
    horizontal_component_weight: float = 1.0
    vertical_component_weight: float = 1.0
    temporal_component_weight: float = 1.0
    huber_delta_vertical: float = 0.1
    coordinate_convention: str = "normalized"
    adaptive: bool = False
    target_horizontal_ratio: float = 0.30
    target_vertical_ratio: float = 0.30
    target_temporal_ratio: float = 0.30
    ema_beta: float = 0.99
    epsilon: float = 1.0e-12
    smoothing: float = 0.05
    update_every: int = 10
    warmup_steps: int = 100
    ramp_steps: int = 1000
    freeze_after_step: int = 0
    component_weight_min: float = 0.0
    component_weight_max: float = 1.0e6

    def validate(self) -> None:
        valid_modes = {
            "none",
            "legacy_diagonal_4d",
            "anisotropic_huber_4d",
            "spatial_hessian_3d",
        }
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported derivative-prior mode: {self.mode}")
        if self.coordinate_convention not in {"normalized", "physical"}:
            raise ValueError("coordinate_convention must be normalized or physical.")
        for name in (
            "weight",
            "horizontal_component_weight",
            "vertical_component_weight",
            "temporal_component_weight",
        ):
            _finite_positive(float(getattr(self, name)), name, allow_zero=True)
        _finite_positive(self.huber_delta_vertical, "huber_delta_vertical")
        if self.mode == "none" and self.weight != 0:
            raise ValueError("Derivative-prior weight must be zero when mode is none.")
        for name in (
            "target_horizontal_ratio",
            "target_vertical_ratio",
            "target_temporal_ratio",
            "epsilon",
            "component_weight_min",
            "component_weight_max",
        ):
            _finite_positive(float(getattr(self, name)), name, allow_zero=name != "epsilon")
        if not 0 <= self.ema_beta < 1 or not 0 < self.smoothing <= 1:
            raise ValueError("ema_beta must be in [0,1) and smoothing in (0,1].")
        if self.update_every <= 0 or min(self.warmup_steps, self.ramp_steps, self.freeze_after_step) < 0:
            raise ValueError("Adaptive-controller intervals cannot be invalid or negative.")
        if self.component_weight_max < self.component_weight_min:
            raise ValueError("component_weight_max must be at least component_weight_min.")
        if self.adaptive and (self.mode == "none" or self.weight == 0):
            raise ValueError("Adaptive weighting requires an active nonzero derivative prior.")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    precision: str = "float32"
    amp: bool = False
    inference_chunk_size: int = 65_536
    diagnostic_probe_size: int = 4096
    history_every: int = 1
    checkpoint_every: int = 1000

    def validate(self) -> None:
        if not (
            self.device in {"auto", "cpu", "cuda"}
            or (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError(f"Invalid device policy: {self.device}")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64.")
        if self.amp and self.precision != "float32":
            raise ValueError("AMP is only supported with the float32 policy.")
        if min(
            self.inference_chunk_size,
            self.diagnostic_probe_size,
            self.history_every,
            self.checkpoint_every,
        ) <= 0:
            raise ValueError("Runtime chunk/probe sizes must be positive.")


@dataclass(frozen=True)
class FourDConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    collocation: CollocationConfig = field(default_factory=CollocationConfig)
    derivative_prior: DerivativePriorConfig = field(default_factory=DerivativePriorConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> "FourDConfig":
        self.model.validate()
        self.optimization.validate()
        self.collocation.validate()
        self.derivative_prior.validate()
        self.runtime.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "optimization": OptimizationConfig,
    "collocation": CollocationConfig,
    "derivative_prior": DerivativePriorConfig,
    "runtime": RuntimeConfig,
}


def config_from_mapping(values: Mapping[str, Any]) -> FourDConfig:
    """Load a fully validated config, rejecting unknown names at every level."""

    if not isinstance(values, Mapping):
        raise ValueError("Configuration must be an object.")
    _only(values, set(_SECTIONS), "root")
    sections: dict[str, Any] = {}
    for name, section_type in _SECTIONS.items():
        supplied = values.get(name, {})
        if not isinstance(supplied, Mapping):
            raise ValueError(f"Configuration section {name} must be an object.")
        allowed = set(section_type.__dataclass_fields__)
        _only(supplied, allowed, name)
        sections[name] = section_type(**supplied)
    return FourDConfig(**sections).validate()


def load_config(path: Path) -> FourDConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_mapping(json.load(handle))


def apply_explicit_overrides(config: FourDConfig, overrides: Mapping[str, Any]) -> FourDConfig:
    """Apply only supplied dotted-name overrides; zero is a valid supplied value."""

    updated = config
    valid: set[str] = set()
    for section_name, section_type in _SECTIONS.items():
        valid.update(f"{section_name}.{name}" for name in section_type.__dataclass_fields__)
    _only(overrides, valid, "overrides")
    for dotted_name, value in overrides.items():
        section_name, field_name = dotted_name.split(".", 1)
        section = getattr(updated, section_name)
        updated = replace(updated, **{section_name: replace(section, **{field_name: value})})
    return updated.validate()
