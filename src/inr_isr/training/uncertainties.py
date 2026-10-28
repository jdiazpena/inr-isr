"""Explicit reported-dNe weighting of scalar density residuals.

The log uncertainty uses first-order propagation, not log10(dNe). No precision
floor, quality filter, weight cap, noise model or confidence claim is inferred.
Existing equal-weight objectives remain owned by training.steps.
"""
import math
import torch

from inr_isr.transforms import affine_inverse
from .steps import validate_data_loss_definition, validate_lineardata_target_scaler


def weighting_definition(method='equal'):
    if method == 'equal':
        return {'data_weighting': 'equal'}
    if method != 'reported_dNe':
        raise ValueError('data_weighting must be equal or reported_dNe')
    return {'data_weighting': method, 'reduction': 'mean_squared_standardized_residual',
            'log_uncertainty': 'first_order_dNe_over_Ne_ln10',
            'invalid_uncertainty': 'fail_without_filter_or_replacement',
            'uncertainty_floor': None, 'weight_cap': None,
            'sum_of_weights_normalization': False}


def _matching_columns(*values):
    reference = values[0]
    if (reference.ndim != 2 or reference.shape[1] != 1 or reference.numel() == 0
            or not reference.is_floating_point()):
        raise ValueError('Uncertainty weighting requires nonempty floating [N,1] columns')
    for value in values:
        if (value.shape != reference.shape or value.device != reference.device
                or value.dtype != reference.dtype or not value.is_floating_point()):
            raise ValueError('Density residuals and uncertainties must have matching shape, dtype and device')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Density residuals and uncertainties must be finite')


def density_uncertainty_sigma(ne, dne, *, data_loss_domain, target_scaler,
                              linear_density_reference=None):
    """Represent reported uncertainty in the units used by the selected residual.

Log: [dNe/(Ne ln10)] times the target's affine normalization scale.
Linear: dNe/reference; the same reference divides the linear residual.
All retained rows must have finite positive Ne and dNe. Large relative errors
are retained; the first-order approximation is explicit rather than repaired.
"""
    validate_data_loss_definition(data_loss_domain, linear_density_reference)
    minimum, maximum = validate_lineardata_target_scaler(target_scaler)
    _matching_columns(ne, dne)
    if not bool((ne > 0).all() & (dne > 0).all()):
        raise ValueError('Reported-dNe weighting requires positive Ne and dNe for every retained row')
    if data_loss_domain == 'log':
        sigma = (dne / ne) * (2. / ((maximum - minimum) * math.log(10.)))
    else:
        sigma = dne / linear_density_reference
    if not bool(torch.isfinite(sigma).all() & (sigma > 0).all()):
        raise ValueError('Converted uncertainty must be finite and positive without clipping')
    return sigma


def standardized_density_mse(prediction, targets, sigma, *, data_loss_domain,
                             target_scaler, linear_density_reference=None):
    """Mean squared residual/sigma, retaining the normalized-log model output."""
    validate_data_loss_definition(data_loss_domain, linear_density_reference)
    minimum, maximum = validate_lineardata_target_scaler(target_scaler)
    _matching_columns(prediction, targets, sigma)
    if not bool((sigma > 0).all()):
        raise ValueError('Converted uncertainty must be positive')
    if data_loss_domain == 'log':
        residual = prediction - targets
    else:
        reference_log = math.log10(linear_density_reference)
        pred = torch.pow(10., affine_inverse(prediction, minimum, maximum) - reference_log)
        truth = torch.pow(10., affine_inverse(targets, minimum, maximum) - reference_log)
        residual = pred - truth
    loss = torch.mean(torch.square(residual / sigma))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite standardized density loss; no automatic clipping or replacement')
    return loss


def uncertainty_data_objective(model, points, targets, sigma, **definition):
    """One forward; the engine owns row alignment and the unchanged update order."""
    validate_data_loss_definition(definition['data_loss_domain'], definition.get('linear_density_reference'))
    validate_lineardata_target_scaler(definition['target_scaler'])
    _matching_columns(targets, sigma)
    if not bool((sigma > 0).all()):
        raise ValueError('Converted uncertainty must be positive before model execution')
    prediction = model(points)
    return prediction, standardized_density_mse(prediction, targets, sigma, **definition)
