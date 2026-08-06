# -*- coding: utf-8 -*-
"""Shared, behavior-stable utilities for synthetic and real-radar INR training.

The training entry points use the same normalized x-y-time mathematics. Keeping these
utilities here prevents the synthetic and radar implementations from silently
changing apart. Derivatives are taken with respect to normalized coordinates; they
are smoothness priors, not a physical PDE.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch


class NormalizedTimeDataset(Protocol):
    """Dataset operations required by shared evaluation and collocation utilities."""

    coord_scalers: dict[str, dict[str, float]]

    def denormalize_target(self, values_norm: np.ndarray) -> np.ndarray:
        """Map normalized predictions back to log10 electron density."""

        ...


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch before any sampling or model initialization."""

    torch.manual_seed(seed)
    np.random.seed(seed)


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    """Append one stable-schema training-history row, creating its header once."""

    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def sample_batch(
    coords: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a full batch or one random without-replacement minibatch."""

    n_samples = coords.shape[0]

    if batch_size <= 0 or batch_size >= n_samples:
        idx = torch.arange(n_samples, device=coords.device)
    else:
        idx = torch.randperm(n_samples, device=coords.device)[:batch_size]

    return coords[idx], values[idx]


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute measured-point residual metrics in the arrays' current units."""

    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    residual = pred - target
    abs_residual = np.abs(residual)

    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_residual))
    bias = float(np.mean(residual))
    max_abs = float(np.max(abs_residual))
    p95_abs = float(np.quantile(abs_residual, 0.95))
    p99_abs = float(np.quantile(abs_residual, 0.99))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "max_abs": max_abs,
        "p95_abs": p95_abs,
        "p99_abs": p99_abs,
    }


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


def safe_ratio(numer: float, denom: float, eps: float = 1e-30) -> float:
    """
    Numerically safe scalar ratio.
    """

    numer = float(numer)
    denom = float(denom)

    if not np.isfinite(numer) or not np.isfinite(denom):
        return float("nan")

    if abs(denom) < eps:
        return float("nan")

    return float(numer / denom)


def make_query_grid_from_points(
    x_km: np.ndarray,
    y_km: np.ndarray,
    nx: int,
    ny: int,
    padding_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a padded Cartesian x-y grid around measured locations."""

    x_min = float(np.min(x_km))
    x_max = float(np.max(x_km))
    y_min = float(np.min(y_km))
    y_max = float(np.max(y_km))

    dx = x_max - x_min
    dy = y_max - y_min

    x_min -= padding_frac * dx
    x_max += padding_frac * dx
    y_min -= padding_frac * dy
    y_max += padding_frac * dy

    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)

    X, Y = np.meshgrid(x_grid, y_grid)

    return X, Y


def estimate_nearest_radius(
    measured_xy: np.ndarray,
    factor: float,
) -> float:
    """Estimate a support radius from median nearest-neighbor beam spacing."""

    measured_xy = np.asarray(measured_xy, dtype=float)

    diff = measured_xy[:, None, :] - measured_xy[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    np.fill_diagonal(dist, np.inf)

    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]

    if nearest.size == 0:
        raise ValueError("Could not estimate nearest-neighbor spacing.")

    median_nearest = float(np.median(nearest))

    return factor * median_nearest


def nearest_distance_mask(
    X: np.ndarray,
    Y: np.ndarray,
    measured_xy: np.ndarray,
    radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark grid points supported by an observation within the given radius."""

    grid_xy = np.column_stack([X.ravel(), Y.ravel()])
    measured_xy = np.asarray(measured_xy, dtype=float)

    diff = grid_xy[:, None, :] - measured_xy[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)

    nearest_dist = np.sqrt(np.min(dist2, axis=1))
    nearest_dist_grid = nearest_dist.reshape(X.shape)

    mask = nearest_dist_grid <= radius_km

    return mask, nearest_dist_grid


def normalize_xy_t_grid_with_dataset(
    dataset: NormalizedTimeDataset,
    X: np.ndarray,
    Y: np.ndarray,
    t_sec: float,
) -> np.ndarray:
    """Normalize physical x-y-time query coordinates with dataset scalers."""

    x_min = dataset.coord_scalers["x_km"]["min"]
    x_max = dataset.coord_scalers["x_km"]["max"]

    y_min = dataset.coord_scalers["y_km"]["min"]
    y_max = dataset.coord_scalers["y_km"]["max"]

    t_min = dataset.coord_scalers["t_sec"]["min"]
    t_max = dataset.coord_scalers["t_sec"]["max"]

    Xn = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
    Yn = 2.0 * (Y - y_min) / (y_max - y_min) - 1.0
    Tn = 2.0 * (float(t_sec) - t_min) / (t_max - t_min) - 1.0

    coords_grid = np.column_stack(
        [
            Xn.ravel(),
            Yn.ravel(),
            np.full(X.size, Tn, dtype=np.float64),
        ]
    ).astype(np.float32)

    return coords_grid


def make_collocation_pool(
    dataset: NormalizedTimeDataset,
    df: pd.DataFrame,
    grid_nx: int,
    grid_ny: int,
    padding_frac: float,
    nearest_radius_factor: float,
) -> tuple[torch.Tensor, float, float, int]:
    """
    Build x-y-t collocation points.

    For each existing time in the current window:
        make the same x-y grid
        mask it by nearest radar-point distance
        assign that time value

    These collocation points do not have data targets.
    They are only used for derivative losses.
    """

    measured_xy = (
        df[["x_km", "y_km"]]
        .drop_duplicates()
        .to_numpy(dtype=float)
    )

    X, Y = make_query_grid_from_points(
        x_km=df["x_km"].to_numpy(dtype=float),
        y_km=df["y_km"].to_numpy(dtype=float),
        nx=grid_nx,
        ny=grid_ny,
        padding_frac=padding_frac,
    )

    nearest_radius_km = estimate_nearest_radius(
        measured_xy,
        factor=nearest_radius_factor,
    )

    valid_mask, _ = nearest_distance_mask(
        X,
        Y,
        measured_xy,
        radius_km=nearest_radius_km,
    )

    xy_valid = np.column_stack([X.ravel(), Y.ravel()])[valid_mask.ravel()]

    unique_times = np.sort(df["t_sec"].unique())

    all_coords = []

    for t_sec in unique_times:
        Xv = xy_valid[:, 0]
        Yv = xy_valid[:, 1]

        coords_t = normalize_xy_t_grid_with_dataset(
            dataset=dataset,
            X=Xv.reshape(-1, 1),
            Y=Yv.reshape(-1, 1),
            t_sec=float(t_sec),
        )

        all_coords.append(coords_t)

    coords_col_np = np.concatenate(all_coords, axis=0).astype(np.float32)

    if coords_col_np.shape[0] == 0:
        raise RuntimeError("No valid collocation points were created.")

    coords_col = torch.from_numpy(coords_col_np)

    valid_fraction = float(valid_mask.mean())

    return coords_col, nearest_radius_km, valid_fraction, int(unique_times.size)


def sample_collocation_points(
    collocation_pool: torch.Tensor,
    num_collocation: int,
) -> torch.Tensor:
    """Sample derivative-evaluation coordinates without replacement."""

    n_total = collocation_pool.shape[0]

    if num_collocation <= 0 or num_collocation >= n_total:
        idx = torch.arange(n_total, device=collocation_pool.device)
    else:
        idx = torch.randperm(n_total, device=collocation_pool.device)[:num_collocation]

    return collocation_pool[idx]


def curvature_losses_xy_t(
    model: torch.nn.Module,
    coords_col: torch.Tensor,
    use_xy: bool,
    use_t: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial x-y curvature and temporal curvature.

    coords_col has columns:
        0: x_norm
        1: y_norm
        2: t_norm

    curv_xy:
        mean(fxx^2 + 2 fxy^2 + fyy^2)

    curv_t:
        mean(ftt^2)

    Derivatives are with respect to normalized coordinates.
    """

    coords_col = coords_col.detach().clone().requires_grad_(True)

    pred = model(coords_col)

    grad = torch.autograd.grad(
        outputs=pred,
        inputs=coords_col,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    zero = pred.new_tensor(0.0)

    curv_xy = zero
    curv_t = zero

    if use_xy:
        fx = grad[:, 0:1]
        fy = grad[:, 1:2]

        grad_fx = torch.autograd.grad(
            outputs=fx,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fx),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_fy = torch.autograd.grad(
            outputs=fy,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fy),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        fxx = grad_fx[:, 0:1]
        fxy = grad_fx[:, 1:2]
        fyy = grad_fy[:, 1:2]

        curv_xy = torch.mean(fxx ** 2 + 2.0 * fxy ** 2 + fyy ** 2)

    if use_t:
        ft = grad[:, 2:3]

        grad_ft = torch.autograd.grad(
            outputs=ft,
            inputs=coords_col,
            grad_outputs=torch.ones_like(ft),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        ftt = grad_ft[:, 2:3]

        curv_t = torch.mean(ftt ** 2)

    return curv_xy, curv_t


def curvature_loss_4d(
    model: torch.nn.Module,
    coords_col: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute 4D spatial-temporal curvature loss using PyTorch autograd:
        L_curvature = mean(f_xx^2 + f_yy^2 + f_zz^2 + f_tt^2)

    coords_col: Tensor of shape (N, 4) with columns (x_norm, y_norm, z_norm, t_norm).
    """
    coords_col = coords_col.detach().clone().requires_grad_(True)
    pred = model(coords_col)

    grad = torch.autograd.grad(
        outputs=pred,
        inputs=coords_col,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    fx = grad[:, 0:1]
    fy = grad[:, 1:2]
    fz = grad[:, 2:3]
    ft = grad[:, 3:4]

    grad_fx = torch.autograd.grad(
        outputs=fx,
        inputs=coords_col,
        grad_outputs=torch.ones_like(fx),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_fy = torch.autograd.grad(
        outputs=fy,
        inputs=coords_col,
        grad_outputs=torch.ones_like(fy),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_fz = torch.autograd.grad(
        outputs=fz,
        inputs=coords_col,
        grad_outputs=torch.ones_like(fz),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_ft = torch.autograd.grad(
        outputs=ft,
        inputs=coords_col,
        grad_outputs=torch.ones_like(ft),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    fxx = grad_fx[:, 0:1]
    fyy = grad_fy[:, 1:2]
    fzz = grad_fz[:, 2:3]
    ftt = grad_ft[:, 3:4]

    curv_xx = torch.mean(fxx ** 2)
    curv_yy = torch.mean(fyy ** 2)
    curv_zz = torch.mean(fzz ** 2)
    curv_tt = torch.mean(ftt ** 2)

    l_curvature = curv_xx + curv_yy + curv_zz + curv_tt

    details = {
        "curv_xx": curv_xx,
        "curv_yy": curv_yy,
        "curv_zz": curv_zz,
        "curv_tt": curv_tt,
        "l_curvature": l_curvature,
    }

    return l_curvature, details



def tensor_near_zero_stats(
    tensor: torch.Tensor | None,
    prefix: str,
    eps: float,
) -> dict[str, float]:
    """
    Compute numerical diagnostics for one derivative tensor.

    The near-zero fraction uses |tensor| < eps.
    The exact-zero fraction uses tensor == 0 exactly and is mostly a bug detector.
    """

    keys = [
        f"{prefix}_rms",
        f"{prefix}_meanabs",
        f"{prefix}_maxabs",
        f"{prefix}_frac_near_zero",
        f"{prefix}_frac_exact_zero",
    ]

    if tensor is None:
        return {key: float("nan") for key in keys}

    x = tensor.detach()

    if x.numel() == 0:
        return {key: float("nan") for key in keys}

    abs_x = x.abs()

    return {
        f"{prefix}_rms": float(torch.sqrt(torch.mean(x ** 2)).item()),
        f"{prefix}_meanabs": float(torch.mean(abs_x).item()),
        f"{prefix}_maxabs": float(torch.max(abs_x).item()),
        f"{prefix}_frac_near_zero": float(torch.mean((abs_x < eps).float()).item()),
        f"{prefix}_frac_exact_zero": float(torch.mean((x == 0.0).float()).item()),
    }


def derivative_diagnostics_xy_t(
    model: torch.nn.Module,
    coords_col: torch.Tensor,
    use_xy: bool,
    use_t: bool,
    zero_eps: float,
) -> dict[str, float]:
    """
    Compute fixed-probe diagnostics for the second derivatives used by the loss.

    This is diagnostic only. It does not change training.

    coords_col has columns:
        0: x_norm
        1: y_norm
        2: t_norm

    Logged components:
        fxx, fxy, fyy, ftt

    Logged aggregate probe losses:
        diag_curv_xy_probe = mean(fxx^2 + 2 fxy^2 + fyy^2)
        diag_curv_t_probe  = mean(ftt^2)
    """

    coords_col = coords_col.detach().clone().requires_grad_(True)

    pred = model(coords_col)

    grad = torch.autograd.grad(
        outputs=pred,
        inputs=coords_col,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    fxx = None
    fxy = None
    fyy = None
    ftt = None

    diag_curv_xy_probe = float("nan")
    diag_curv_t_probe = float("nan")

    if use_xy:
        fx = grad[:, 0:1]
        fy = grad[:, 1:2]

        grad_fx = torch.autograd.grad(
            outputs=fx,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fx),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_fy = torch.autograd.grad(
            outputs=fy,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fy),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        fxx = grad_fx[:, 0:1]
        fxy = grad_fx[:, 1:2]
        fyy = grad_fy[:, 1:2]

        diag_curv_xy_probe = float(
            torch.mean(fxx ** 2 + 2.0 * fxy ** 2 + fyy ** 2).detach().item()
        )

    if use_t:
        ft = grad[:, 2:3]

        grad_ft = torch.autograd.grad(
            outputs=ft,
            inputs=coords_col,
            grad_outputs=torch.ones_like(ft),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        ftt = grad_ft[:, 2:3]

        diag_curv_t_probe = float(torch.mean(ftt ** 2).detach().item())

    out: dict[str, float] = {
        "diag_curv_xy_probe": diag_curv_xy_probe,
        "diag_curv_t_probe": diag_curv_t_probe,
    }

    out.update(tensor_near_zero_stats(fxx, "fxx", zero_eps))
    out.update(tensor_near_zero_stats(fxy, "fxy", zero_eps))
    out.update(tensor_near_zero_stats(fyy, "fyy", zero_eps))
    out.update(tensor_near_zero_stats(ftt, "ftt", zero_eps))

    return out


def parameter_grad_norm_from_loss(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> float:
    """
    L2 norm of d(loss)/d(theta).

    This does not populate parameter .grad fields because it uses autograd.grad.
    """

    if not isinstance(loss, torch.Tensor):
        return float("nan")

    if not loss.requires_grad:
        return 0.0

    grads = torch.autograd.grad(
        outputs=loss,
        inputs=parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )

    total = 0.0

    for grad in grads:
        if grad is None:
            continue

        total += float(torch.sum(grad.detach() ** 2).item())

    return float(np.sqrt(total))


def parameter_grad_norm_from_existing_grads(
    parameters: list[torch.nn.Parameter],
) -> float:
    """
    L2 norm of the gradients currently stored in parameter .grad fields.
    """

    total = 0.0

    for param in parameters:
        if param.grad is None:
            continue

        total += float(torch.sum(param.grad.detach() ** 2).item())

    return float(np.sqrt(total))


@torch.no_grad()
def evaluate_model_on_coords(
    model: torch.nn.Module,
    coords_np: np.ndarray,
    dataset: NormalizedTimeDataset,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate normalized coordinates in chunks and return log10 density."""

    model.eval()

    outputs = []
    n = coords_np.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        coords_chunk = torch.from_numpy(coords_np[start:end]).to(device)
        pred_chunk = model(coords_chunk).detach().cpu().numpy()

        outputs.append(pred_chunk)

    pred_norm = np.concatenate(outputs, axis=0)
    pred_log10 = dataset.denormalize_target(pred_norm)

    return pred_log10[:, 0]


def select_plot_time_indices(
    df: pd.DataFrame,
    num_plot_times: int,
) -> list[int]:
    """Choose existing time indices spanning the start, middle, and end."""

    unique_times = np.sort(df["time_index"].unique())

    if num_plot_times <= 0:
        return []

    if num_plot_times >= unique_times.size:
        return [int(x) for x in unique_times]

    picks = np.linspace(0, unique_times.size - 1, num_plot_times)
    picks = np.round(picks).astype(int)

    return [int(unique_times[i]) for i in picks]
