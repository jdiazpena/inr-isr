"""Shared xyt analyzer mechanics, without choosing a truth or error convention.

Both existing analyzers call these exact numerical operations. Their native
modules still choose instantaneous versus integration-averaged truth, physical
density versus log-space error, plot labels, output schema and public settings.
This module does not fit scalers, select data, save files or alter model mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .transforms import denormalize_target


def normalized_gradient_chunks(
    model: torch.nn.Module,
    coordinates: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
    policy: str,
    dtype: torch.dtype | None = None,
):
    """Yield ordered normalized predictions and input gradients, one chunk each.

    The xyt analyzer historically uses from_numpy, a detached clone and an
    explicit ones-like autograd seed. The xyzt truth evaluator uses as_tensor at
    its declared dtype and a summed-output seed. These are explicit numerical
    policies, not interchangeable implementation details. Native callers still
    own model mode, physical chain factors, inverse scale and output shapes.
    """
    if policy not in {"xyt", "xyzt"}:
        raise ValueError(f"Unknown normalized-gradient policy: {policy}")
    for start in range(0, len(coordinates), chunk_size):
        end = min(start + chunk_size, len(coordinates))
        if policy == "xyt":
            points = torch.from_numpy(coordinates[start:end]).to(device)
            points = points.detach().clone().requires_grad_(True)
            prediction = model(points)
            gradient = torch.autograd.grad(
                outputs=prediction,
                inputs=points,
                grad_outputs=torch.ones_like(prediction),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
        else:
            points = torch.as_tensor(
                coordinates[start:end], dtype=dtype, device=device
            ).requires_grad_(True)
            prediction = model(points)
            gradient = torch.autograd.grad(prediction.sum(), points)[0]
        yield prediction, gradient


def predict_log10_and_gradients_on_coords(
    model: torch.nn.Module,
    coords_norm: np.ndarray,
    coord_scalers: dict,
    target_scaler: dict,
    device: torch.device,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Return xyt log10 density and its x/y/t physical-coordinate derivatives.

    The network differentiates normalized log10(Ne) against normalized x,y,t.
    Each physical derivative multiplies the target half-range and the relevant
    2/coordinate-span chain factor. Preserve NumPy multiplication order and the
    original torch from_numpy/clone/autograd calls for exact old-model replay.
    """
    pred_list = []
    dx_list = []
    dy_list = []
    dt_list = []

    target_scale = 0.5 * (float(target_scaler["max"]) - float(target_scaler["min"]))

    dxnorm_dx = 2.0 / (float(coord_scalers["x_km"]["max"]) - float(coord_scalers["x_km"]["min"]))
    dynorm_dy = 2.0 / (float(coord_scalers["y_km"]["max"]) - float(coord_scalers["y_km"]["min"]))
    dtnorm_dt = 2.0 / (float(coord_scalers["t_sec"]["max"]) - float(coord_scalers["t_sec"]["min"]))

    for pred_norm, grad_norm in normalized_gradient_chunks(
        model, coords_norm, device=device, chunk_size=chunk_size, policy="xyt"
    ):
        pred_log10 = denormalize_target(
            pred_norm.detach().cpu().numpy()[:, 0],
            target_scaler=target_scaler,
        )
        pred_list.append(pred_log10)
        dx_list.append((target_scale * grad_norm[:, 0].detach().cpu().numpy() * dxnorm_dx).astype(np.float64))
        dy_list.append((target_scale * grad_norm[:, 1].detach().cpu().numpy() * dynorm_dy).astype(np.float64))
        dt_list.append((target_scale * grad_norm[:, 2].detach().cpu().numpy() * dtnorm_dt).astype(np.float64))

    return {
        "pred_log10_Ne": np.concatenate(pred_list, axis=0),
        "pred_dlog10Ne_dx_km": np.concatenate(dx_list, axis=0),
        "pred_dlog10Ne_dy_km": np.concatenate(dy_list, axis=0),
        "pred_dlog10Ne_dt_sec": np.concatenate(dt_list, axis=0),
    }


def compute_error_metrics(error: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Keep both analyzers' seven-column reduction and cast order unchanged."""
    error = np.asarray(error, dtype=np.float64)
    abs_error = np.abs(error)
    return {
        f"{prefix}mse": float(np.mean(error ** 2)),
        f"{prefix}rmse": float(np.sqrt(np.mean(error ** 2))),
        f"{prefix}mae": float(np.mean(abs_error)),
        f"{prefix}bias": float(np.mean(error)),
        f"{prefix}max_abs": float(np.max(abs_error)),
        f"{prefix}p95_abs": float(np.quantile(abs_error, 0.95)),
        f"{prefix}p99_abs": float(np.quantile(abs_error, 0.99)),
    }


def select_analyzer_time_indices(
    obs_df: pd.DataFrame, selection: str | list[int], unsupported_label: str,
) -> list[int]:
    """Select existing records; caller owns its historical invalid-option text."""
    available = [int(x) for x in np.sort(obs_df["time_index"].unique())]
    if isinstance(selection, list):
        requested = [int(x) for x in selection]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested time_index values are unavailable: {missing}")
        return requested
    if selection == "all":
        return available
    if selection == "first_middle_last":
        if len(available) <= 3:
            return available
        return [available[0], available[len(available) // 2], available[-1]]
    raise ValueError(f"Unsupported {unsupported_label}: {selection}")


def make_analyzer_dense_grid(
    config: dict, coord_scalers: dict, grid_nx: int, grid_ny: int,
    domain_mode: str, unsupported_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the old float64 full-domain/training-extent mesh, in x then y order."""
    if domain_mode == "full_domain":
        half = 0.5 * float(config["domain_size_km"])
        x = np.linspace(-half, half, int(grid_nx), dtype=np.float64)
        y = np.linspace(-half, half, int(grid_ny), dtype=np.float64)
    elif domain_mode == "training_extent":
        x = np.linspace(
            float(coord_scalers["x_km"]["min"]),
            float(coord_scalers["x_km"]["max"]),
            int(grid_nx),
            dtype=np.float64,
        )
        y = np.linspace(
            float(coord_scalers["y_km"]["min"]),
            float(coord_scalers["y_km"]["max"]),
            int(grid_ny),
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported {unsupported_label}: {domain_mode}")
    X, Y = np.meshgrid(x, y)
    return X, Y
