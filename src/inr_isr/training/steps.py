"""Explicit data/loss expressions and ordered parameter-update orchestration.

Recipes, not dimensions or filenames, select the old scientific policy. Reader,
prior, controller, checkpoint and history stages stay at their native boundaries.
These shared operations do not introduce a scheduler, dropout or physical law.
"""
from __future__ import annotations
from collections.abc import Callable,Sequence
import math
from numbers import Real
from typing import Literal
import torch
import torch.nn.functional as functional
from inr_isr.transforms import affine_inverse
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


def validate_data_loss_definition(data_loss_domain='log', linear_density_reference=None):
    """Resolve explicit data-loss units without fitting scales or changing weights.

    Missing historical selectors mean the original normalized-log MSE. Linear
    fitting requires a declared fixed density reference in m^-3; it is a common
    numerical unit, not an observation-dependent uncertainty or weight.
    """
    if data_loss_domain not in ('log', 'linear'):
        raise ValueError("data_loss_domain must be 'log' or 'linear'")
    if data_loss_domain == 'log':
        if linear_density_reference is not None:
            raise ValueError('Log data loss requires linear_density_reference=None')
        reference = None
    else:
        if (isinstance(linear_density_reference, bool)
                or not isinstance(linear_density_reference, Real)
                or not math.isfinite(linear_density_reference)
                or linear_density_reference <= 0):
            raise ValueError('Linear data loss requires a finite positive linear_density_reference in m^-3')
        reference = float(linear_density_reference)
    return {'data_loss_domain': data_loss_domain, 'linear_density_reference': reference}


def validate_lineardata_target_scaler(target_scaler, *, normalize_values=True):
    """Require the selected-window normalized log10 target and a valid inverse.

    This deliberately accepts the3D scalar min/max schema, not the serialized4D
    scaler. No clipping, replacement limits or inferred target units are allowed.
    """
    if normalize_values is not True:
        raise ValueError('Linear data loss requires normalized log10_Ne targets')
    if not isinstance(target_scaler, dict) or target_scaler.get('column') != 'log10_Ne':
        raise ValueError('Linear data loss requires a log10_Ne target scaler')
    minimum, maximum = target_scaler.get('min'), target_scaler.get('max')
    if (isinstance(minimum, bool) or isinstance(maximum, bool)
            or not isinstance(minimum, Real) or not isinstance(maximum, Real)
            or not math.isfinite(minimum) or not math.isfinite(maximum)
            or maximum <= minimum or not math.isfinite(maximum - minimum)):
        raise ValueError('Linear data loss requires finite target min/max with max > min')
    return float(minimum), float(maximum)


def density_mse_lineardata(prediction: torch.Tensor, targets: torch.Tensor, *,
                          target_scaler, linear_density_reference,
                          normalize_values=True) -> torch.Tensor:
    """Equally weighted density MSE, retaining normalized-log network tensors.

    First invert the saved target affine map to physical log10(Ne/m^-3), then
    exponentiate in the declared common reference unit. The residual is never
    logged again. Inputs retain dtype/device and gradients; no detached NumPy
    conversion, clipping, observation weights, RNG or optimizer is involved.
    """
    definition = validate_data_loss_definition('linear', linear_density_reference)
    minimum, maximum = validate_lineardata_target_scaler(
        target_scaler, normalize_values=normalize_values)
    if (prediction.shape != targets.shape or prediction.ndim != 2
            or prediction.shape[1] != 1 or prediction.numel() == 0):
        raise ValueError('Linear data loss requires matching nonempty [N,1] tensors')
    if (not prediction.is_floating_point() or not targets.is_floating_point()
            or prediction.dtype != targets.dtype or prediction.device != targets.device):
        raise ValueError('Linear data loss requires matching floating tensor dtype and device')
    if not bool(torch.isfinite(prediction).all() & torch.isfinite(targets).all()):
        raise FloatingPointError('Nonfinite linear data loss inputs; predictions and targets must be finite')
    reference_log10 = math.log10(definition['linear_density_reference'])
    prediction_log10 = affine_inverse(prediction, minimum, maximum)
    target_log10 = affine_inverse(targets, minimum, maximum)
    prediction_density = torch.pow(10.0, prediction_log10 - reference_log10)
    target_density = torch.pow(10.0, target_log10 - reference_log10)
    loss = functional.mse_loss(prediction_density, target_density)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite linear data loss; predictions are not clipped or rescaled automatically')
    return loss


def data_objective_lineardata(model, points: torch.Tensor, targets: torch.Tensor, *,
                             target_scaler, linear_density_reference,
                             normalize_values=True) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward followed by linear-density MSE; return normalized-log output.

    The existing log objective and native3D/4D execution recipes remain unchanged.
    Configuration errors fail before the callable is evaluated.
    """
    validate_data_loss_definition('linear', linear_density_reference)
    validate_lineardata_target_scaler(target_scaler, normalize_values=normalize_values)
    prediction = model(points)
    loss = density_mse_lineardata(prediction, targets, target_scaler=target_scaler,
                                linear_density_reference=linear_density_reference,
                                normalize_values=normalize_values)
    return prediction, loss


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
