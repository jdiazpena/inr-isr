"""Reference-ratio adaptive weights for named derivative-prior components."""

from __future__ import annotations

from dataclasses import dataclass, field

import math

from .config import DerivativePriorConfig


COMPONENTS = ("horizontal", "vertical", "temporal")


from inr_isr.training.controllers import PostUpdateReferenceRatioController


@dataclass
class ReferenceRatioController(PostUpdateReferenceRatioController):
    """Keep the public xyzt class path/state while executing the shared policy."""

    def update(self, step: int, data_loss: float, components: dict[str, float]) -> dict[str, float]:
        """Native post-Adam observation boundary; no extra updates or state changes."""
        return super().update(step, data_loss, components)
