"""Optional linear-density spatial curvature; original log kernel stays intact."""
import math
import torch
from inr_isr.transforms import affine_inverse
from .derivatives import coordinate_gradient, hessian_rows


def spatial_curvature_definition(domain='log', density_reference=None):
    if domain not in ('log', 'linear'):
        raise ValueError('spatial_curvature_domain must be log or linear')
    if domain == 'log':
        if density_reference is not None:
            raise ValueError('Log curvature requires linear_curvature_density_reference=None')
    elif (density_reference is None or isinstance(density_reference, bool)
          or not math.isfinite(float(density_reference)) or float(density_reference) <= 0):
        raise ValueError('Linear curvature requires a positive fixed density reference in m^-3')
    return {'spatial_curvature_domain': domain,
            'linear_curvature_density_reference': float(density_reference) if density_reference is not None else None}


def add_spatial_curvature_arguments(parser):
    parser.add_argument('--spatial_curvature_domain', choices=['log', 'linear'], default='log',
                        help='Spatial Hessian variable; log preserves the original method.')
    parser.add_argument('--linear_curvature_density_reference', type=float, default=None,
                        help='One fixed Ne reference in m^-3 for linear spatial curvature only.')


def curvature_losses_xy_t_linear(model, coords_col, use_xy, use_t, *, target_scaler, density_reference):
    """Full xy Hessian of Ne/N0 with normalized coordinates, including chain rule.

    This option changes spatial curvature only. Temporal regularization is a
    separate study; reject it rather than quietly changing its density domain.
    """
    spatial_curvature_definition('linear', density_reference)
    if use_t:
        raise ValueError('Linear spatial-curvature comparison requires temporal training off')
    low, high = float(target_scaler['min']), float(target_scaler['max'])
    if target_scaler.get('column') != 'log10_Ne' or not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError('Linear curvature requires a finite normalized log10_Ne target scaler')
    points = coords_col.detach().clone().requires_grad_(True)
    log_density = affine_inverse(model(points), low, high)
    density = torch.pow(10., log_density - math.log10(density_reference))
    zero = density.new_tensor(0.)
    if not use_xy:
        return zero, zero
    first = coordinate_gradient(density, points, only_inputs=True)
    rows = hessian_rows(first, points, axes=('x', 'y', 't'), selected=('x', 'y'), only_inputs=True)
    xx, xy, yy = rows['x'][:, 0:1], rows['x'][:, 1:2], rows['y'][:, 1:2]
    spatial = torch.mean(xx**2 + 2.*xy**2 + yy**2)
    if not bool(torch.isfinite(spatial)):
        raise FloatingPointError('Nonfinite linear-density curvature; no clipping or fallback')
    return spatial, zero
