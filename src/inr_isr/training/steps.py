"""Explicit data/loss expressions and ordered parameter-update orchestration.

Recipes, not dimensions or filenames, select the old scientific policy. Reader,
prior, controller, checkpoint and history stages stay at their native boundaries.
These shared operations do not introduce a scheduler, dropout or physical law.
"""
from __future__ import annotations
from collections.abc import Callable,Sequence
from typing import Literal
import torch
import torch.nn.functional as functional
Recipe=Literal['xyt','xyzt']


def data_objective(model: torch.nn.Module, points: torch.Tensor, targets: torch.Tensor, *,
                   recipe: Recipe) -> tuple[torch.Tensor,torch.Tensor]:
    """One original forward and normalized log-density MSE; return prediction/loss.

points [N,D] and targets [N,1] retain caller dtype/device. xyt keeps F.mse_loss;
xyzt keeps mean((prediction-target)**2) inside the caller's existing AMP context.
The caller controls model.train and gradient clearing. Unknown recipes fail before
forward; no source/dimension inference, coercion, random draw or extra model call.
"""
    if recipe not in ('xyt','xyzt'):
        raise ValueError(f'Unknown native training recipe: {recipe}')
    prediction=model(points)
    loss=functional.mse_loss(prediction,targets) if recipe=='xyt' else torch.mean((prediction-targets)**2)
    return prediction,loss


def compose_xyt_loss(data_loss: torch.Tensor, weighted_xy: torch.Tensor,
                     weighted_t: torch.Tensor) -> torch.Tensor:
    """Retain original left-associated data+weighted_xy+weighted_t scalar expression.

Weights and curvature are already evaluated in native order. No detached copies,
reduction reassociation or premature controller update: this graph is backpropagated
once by the whole-loss xyt recipe after its pre-Adam best-save decisions.
"""
    return data_loss+weighted_xy+weighted_t


def run_operations(operations: Sequence[tuple[str,Callable[[],object]]]) -> None:
    """Execute named operations in supplied order, propagating the first failure.

Names document actual optimizer-boundary stages and support separate test-only
observation. No retry, suppression, RNG, context/device switch or side operation.
Both native recipes use this same orchestrator with distinct explicit sequences.
"""
    for name,operation in operations:
        operation()


def parameter_update(optimizer: torch.optim.Optimizer, *, recipe: Recipe,
                     total_loss: torch.Tensor | None = None, scaler=None,
                     compute_grad_norm: bool = False, parameters=(), diagnostics=None,
                     grad_norm: Callable | None = None) -> None:
    """Apply the original named recipe's update stages, mutating gradients/weights.

xyt: clear(set_to_none), whole-loss backward, optional existing-grad norm written
to the native diagnostics dictionary, Adam. Component-gradient probes have already
run and best checkpoints already saved. xyzt: scaler.step then scaler.update;
native data/prior/constraint microbatch backwards and unscale/norm have already run.
No new clipping/scaler/config validation changes are applied to valid native calls.
"""
    if recipe=='xyt':
        def existing_gradient_probe():
            """Write only the already-requested native post-backward norm field."""
            if compute_grad_norm:
                diagnostics['grad_norm_total']=grad_norm(parameters=parameters)
        operations=(('zero_grad',lambda:optimizer.zero_grad(set_to_none=True)),
                    ('backward',lambda:total_loss.backward()),
                    ('gradient_probe',existing_gradient_probe),
                    ('optimizer',lambda:optimizer.step()))
    elif recipe=='xyzt':
        operations=(('optimizer',lambda:scaler.step(optimizer)),
                    ('scaler_update',lambda:scaler.update()))
    else:
        raise ValueError(f'Unknown native training recipe: {recipe}')
    run_operations(operations)
