"""Explicit distinct pre-update xyt and post-update xyzt controller policies.

Sharing a home does not imply equal formulas or timing. xyt uses a stabilized
data reference/floor/separate caps; xyzt uses component epsilon/common bounds.
Native trainers place these policies at their original optimizer boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import math
import numpy as np
from inr_isr_4d.config import DerivativePriorConfig
COMPONENTS = ("horizontal", "vertical", "temporal")

def ramp_weight(
    target_weight: float,
    step: int,
    num_steps: int,
    ramp_frac: float,
) -> float:
    """Linearly introduce a regularization weight during the initial ramp."""

    if target_weight <= 0.0:
        return 0.0

    if ramp_frac <= 0.0:
        return float(target_weight)

    ramp_steps = max(1, int(ramp_frac * num_steps))
    factor = min(1.0, step / ramp_steps)

    return float(target_weight * factor)

def update_ema_scalar(
    old_value: float | None,
    new_value: float,
    beta: float,
) -> float:
    """
    Exponential moving average for scalar diagnostics.

    beta close to 1 gives a slow/stable average.
    """

    new_value = float(new_value)

    if old_value is None or not np.isfinite(old_value):
        return new_value

    return float(beta * old_value + (1.0 - beta) * new_value)

def clamp_float(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a float to [min_value, max_value].
    """

    value = float(value)
    min_value = float(min_value)
    max_value = float(max_value)

    if max_value < min_value:
        raise ValueError("max_value must be >= min_value")

    return float(min(max(value, min_value), max_value))

def update_xyt_weights(args, *, step, use_xy_curv, use_t_curv, data_scalar_current, curv_xy_scalar_current, curv_t_scalar_current, data_loss_ema, curv_xy_raw_ema, curv_t_raw_ema, lambda_curv_xy_base, lambda_curv_t_base, lambda_curv_xy_target, lambda_curv_t_target, lambda_frozen):
    """Apply the existing xyt pre-backward controller and return named scalar state.

    Inputs are normalized MSE/curvature floats plus previous EMA/base/target/freeze
    state and the unchanged CLI namespace. EMAs update even in fixed mode. Reference
    targets use the stabilized data floor, separate curvature floor/caps, strict
    step>warmup eligibility, monotone freeze, then smoothing and one-based ramp.
    Return all original names so each native history/checkpoint/observer keeps its
    schema. No tensor operations, RNG draws or optimizer update occur here.
    """
    data_loss_ema = update_ema_scalar(
        old_value=data_loss_ema,
        new_value=data_scalar_current,
        beta=args.loss_ema_beta,
    )

    curv_xy_raw_ema = update_ema_scalar(
        old_value=curv_xy_raw_ema,
        new_value=curv_xy_scalar_current,
        beta=args.loss_ema_beta,
    )

    curv_t_raw_ema = update_ema_scalar(
        old_value=curv_t_raw_ema,
        new_value=curv_t_scalar_current,
        beta=args.loss_ema_beta,
    )

    data_reference = max(float(data_loss_ema), float(args.epsilon_data))

    # --------------------------------------------------------
    # Reference-ratio lambda update.
    #
    # The controller targets weighted curvature terms relative to
    # data_reference, not relative to the collapsing instantaneous
    # data loss.
    # --------------------------------------------------------
    lambda_update_active = False

    if args.reference_loss_weights:
        if use_xy_curv and curv_xy_raw_ema > args.curvature_ema_floor:
            lambda_curv_xy_target = (
                args.target_xy_ratio * data_reference / curv_xy_raw_ema
            )
            lambda_curv_xy_target = clamp_float(
                lambda_curv_xy_target,
                args.lambda_curv_xy_min,
                args.lambda_curv_xy_max,
            )
        else:
            lambda_curv_xy_target = 0.0

        if use_t_curv and curv_t_raw_ema > args.curvature_ema_floor:
            lambda_curv_t_target = (
                args.target_t_ratio * data_reference / curv_t_raw_ema
            )
            lambda_curv_t_target = clamp_float(
                lambda_curv_t_target,
                args.lambda_curv_t_min,
                args.lambda_curv_t_max,
            )
        else:
            lambda_curv_t_target = 0.0

        if args.freeze_lambdas_after_step > 0 and step >= args.freeze_lambdas_after_step:
            lambda_frozen = True

        can_update = (
            step > args.lambda_warmup_steps
            and not lambda_frozen
            and (step % args.lambda_update_every == 0 or step == 1)
        )

        if can_update:
            lambda_update_active = True

            s = float(args.lambda_smoothing)

            lambda_curv_xy_base = (
                (1.0 - s) * lambda_curv_xy_base
                + s * lambda_curv_xy_target
            )

            lambda_curv_t_base = (
                (1.0 - s) * lambda_curv_t_base
                + s * lambda_curv_t_target
            )
    else:
        lambda_curv_xy_base = float(args.lambda_curv_xy)
        lambda_curv_t_base = float(args.lambda_curv_t)
        lambda_curv_xy_target = float(args.lambda_curv_xy)
        lambda_curv_t_target = float(args.lambda_curv_t)

    lambda_curv_xy_eff = ramp_weight(
        target_weight=lambda_curv_xy_base,
        step=step,
        num_steps=args.num_steps,
        ramp_frac=args.reg_ramp_frac,
    )

    lambda_curv_t_eff = ramp_weight(
        target_weight=lambda_curv_t_base,
        step=step,
        num_steps=args.num_steps,
        ramp_frac=args.reg_ramp_frac,
    )

    return {
        'data_loss_ema': data_loss_ema,
        'curv_xy_raw_ema': curv_xy_raw_ema,
        'curv_t_raw_ema': curv_t_raw_ema,
        'lambda_curv_xy_base': lambda_curv_xy_base,
        'lambda_curv_t_base': lambda_curv_t_base,
        'lambda_curv_xy_target': lambda_curv_xy_target,
        'lambda_curv_t_target': lambda_curv_t_target,
        'lambda_frozen': lambda_frozen,
        'data_reference': data_reference,
        'lambda_update_active': lambda_update_active,
        'lambda_curv_xy_eff': lambda_curv_xy_eff,
        'lambda_curv_t_eff': lambda_curv_t_eff,
    }

@dataclass
class PostUpdateReferenceRatioController:
    config: DerivativePriorConfig
    weights: dict[str, float] = field(default_factory=dict)
    data_ema: float | None = None
    component_ema: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Retain configured starting weights and existing empty-state semantics."""
        if not self.weights:
            self.weights = {
                "horizontal": self.config.horizontal_component_weight,
                "vertical": self.config.vertical_component_weight,
                "temporal": self.config.temporal_component_weight,
            }

    def update(self, step: int, data_loss: float, components: dict[str, float]) -> dict[str, float]:
        """Update after Adam: existing EMA, inclusive warmup/freeze, caps and ramp."""
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
        """Return copied native weights/EMA state for full continuation."""
        return {
            "weights": dict(self.weights),
            "data_ema": self.data_ema,
            "component_ema": dict(self.component_ema),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore native scalar coercions and field names without validation changes."""
        self.weights = {name: float(value) for name, value in dict(state["weights"]).items()}
        self.data_ema = None if state["data_ema"] is None else float(state["data_ema"])
        self.component_ema = {
            name: float(value) for name, value in dict(state["component_ema"]).items()
        }
