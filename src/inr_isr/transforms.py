"""Affine arithmetic with explicit legacy coercion and fitting adapters.

Physical columns are x/y[/z] in km and time in seconds; targets are log10(Ne/m^-3).
The kernels do not fit, clip, change dtype or choose rows. Selected-window xyt and
training-split xyzt keep their own fit/validation and serialized scaler policies.
"""
from __future__ import annotations
import numpy as np


def affine_forward(values, minimum, maximum):
    """Map u to2*(u-min)/(max-min)-1 with the historical operation order.

Inputs are scalar or broadcast-compatible numeric arrays; caller owns validation,
dtype/coercion and fitting population. Return a new arithmetic result, no mutation,
clipping, RNG or unit changes beyond normalization. Held-out values may exceed1.
"""
    return 2.0 * (values - minimum) / (maximum - minimum) - 1.0


def affine_inverse(values, minimum, maximum):
    """Invert normalized scalar/array columns, retaining dtype promotion/order.

Return physical coordinate units or physical log density, NOT linear density.
No validation, coercion, clipping, mutation or randomness is added by this kernel.
"""
    return 0.5 * (values + 1.0) * (maximum - minimum) + minimum


def normalize_minus1_plus1(values: np.ndarray, vmin: float | None = None,
                          vmax: float | None = None, name: str = "values"):
    """Legacy xyt selected-window map: float64 and nanmin/nanmax or supplied bounds.

Return(normalized array, min, max); shape is retained. Both datasets use this
same kernel. NaNs/warnings and limit-error order intentionally match old code;
this is not the finite-only training-split4D fitting policy. No mutation or RNG.
"""
    values = np.asarray(values, dtype=np.float64)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"{name}: non-finite normalization limits.")
    if vmax <= vmin:
        raise ValueError(f"{name}: cannot normalize because vmax <= vmin "
                         f"({vmax} <= {vmin}).")
    return affine_forward(values, vmin, vmax), vmin, vmax


def denormalize_minus1_plus1(values_norm: np.ndarray, vmin: float, vmax: float):
    """Legacy xyt inverse: coerce any shape tofloat64 then return original units.

No limit validation is added: callers retain old scalar state/error conventions.
The target inverse returns log10(Ne/m^-3), not10**logdensity. No mutation/RNG.
"""
    values_norm = np.asarray(values_norm, dtype=np.float64)
    return affine_inverse(values_norm, vmin, vmax)


def denormalize_target(pred_norm: np.ndarray, target_scaler: dict):
    """Restore3D analyzer predictions using saved scalar min/max checkpoint state.

Any input shape becomesfloat64, then physical logdensity. Required dictionary
keys/coercion/error order remain the legacy contract; no clipping or fitting.
"""
    values_norm = np.asarray(pred_norm, dtype=np.float64)
    vmin = float(target_scaler["min"])
    vmax = float(target_scaler["max"])
    return affine_inverse(values_norm, vmin, vmax)


def normalize_coords_from_scalers(x_km: np.ndarray, y_km: np.ndarray,
                                 t_sec: np.ndarray, coord_scalers: dict):
    """Flatten physical x/y/time arrays tofloat32[N,3] using saved xyt bounds.

Read all six limits in original order before arithmetic. Do not coerce each input
tofloat64 first: its original dtype affects rounding. No clipping/fit/RNG/mutation.
"""
    x_min = float(coord_scalers["x_km"]["min"])
    x_max = float(coord_scalers["x_km"]["max"])
    y_min = float(coord_scalers["y_km"]["min"])
    y_max = float(coord_scalers["y_km"]["max"])
    t_min = float(coord_scalers["t_sec"]["min"])
    t_max = float(coord_scalers["t_sec"]["max"])
    x_norm = affine_forward(x_km, x_min, x_max)
    y_norm = affine_forward(y_km, y_min, y_max)
    t_norm = affine_forward(t_sec, t_min, t_max)
    return np.column_stack([x_norm.ravel(), y_norm.ravel(), t_norm.ravel()]).astype(np.float32)
