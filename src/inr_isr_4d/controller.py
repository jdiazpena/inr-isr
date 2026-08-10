"""Reference-ratio adaptive weights for named derivative-prior components."""

from __future__ import annotations

from dataclasses import dataclass, field

import math

from .config import DerivativePriorConfig


COMPONENTS = ("horizontal", "vertical", "temporal")


@dataclass
class ReferenceRatioController:
    config: DerivativePriorConfig
    weights: dict[str, float] = field(default_factory=dict)
    data_ema: float | None = None
    component_ema: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = {
                "horizontal": self.config.horizontal_component_weight,
                "vertical": self.config.vertical_component_weight,
                "temporal": self.config.temporal_component_weight,
            }

    def update(self, step: int, data_loss: float, components: dict[str, float]) -> dict[str, float]:
        beta = self.config.ema_beta
        self.data_ema = data_loss if self.data_ema is None else beta * self.data_ema + (1 - beta) * data_loss
        for name in COMPONENTS:
            value = float(components[name])
            old = self.component_ema.get(name)
            self.component_ema[name] = value if old is None else beta * old + (1 - beta) * value

        eligible = (
            self.config.adaptive
            and step >= self.config.warmup_steps
            and step % self.config.update_every == 0
            and not (self.config.freeze_after_step and step >= self.config.freeze_after_step)
        )
        if not eligible:
            return dict(self.weights)

        ramp = 1.0 if self.config.ramp_steps == 0 else min(1.0, (step + 1) / self.config.ramp_steps)
        targets = {
            "horizontal": self.config.target_horizontal_ratio,
            "vertical": self.config.target_vertical_ratio,
            "temporal": self.config.target_temporal_ratio,
        }
        for name in COMPONENTS:
            raw = targets[name] * float(self.data_ema) / (self.component_ema[name] + self.config.epsilon)
            raw *= ramp
            raw = min(max(raw, self.config.component_weight_min), self.config.component_weight_max)
            old = self.weights[name]
            value = (1 - self.config.smoothing) * old + self.config.smoothing * raw
            if not math.isfinite(value):
                raise RuntimeError(f"Adaptive {name} weight became non-finite.")
            self.weights[name] = value
        return dict(self.weights)

    def state_dict(self) -> dict[str, object]:
        return {
            "weights": dict(self.weights),
            "data_ema": self.data_ema,
            "component_ema": dict(self.component_ema),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.weights = {name: float(value) for name, value in dict(state["weights"]).items()}
        self.data_ema = None if state["data_ema"] is None else float(state["data_ema"])
        self.component_ema = {
            name: float(value) for name, value in dict(state["component_ema"]).items()
        }
