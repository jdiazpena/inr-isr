# -*- coding: utf-8 -*-
"""
synthetic_analyze_reconstruction.py

Dense error analysis for ONE trained synthetic INR run.

This script is intentionally configured from the top of the file, not from
command-line arguments. The training scripts are command-line driven because
we will launch many runs from bash. This analysis script is usually run once
for one completed training folder, so the paths/settings are kept here.

What it does:
    1. Loads one trained model checkpoint.
    2. Reconstructs the analytical synthetic truth from synthetic_config.json.
    3. Evaluates truth and INR prediction on a dense x-y grid at selected times.
    4. Saves density error maps and CSV summaries.
    5. Optionally computes first-gradient errors using autograd.

Run from inside your synthetic project folder, for example:
    cd ~/postdoc/codex-inr-radar/inf_fakedata_3d
    python src/synthetic_analyze_reconstruction.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from inr_isr.checkpoints import (torch_load_checkpoint as _torch_load_checkpoint,
                                 build_model_from_checkpoint as _build_model_from_checkpoint)
from inr_isr.transforms import (denormalize_target as _denormalize_target,
                                normalize_coords_from_scalers as _normalize_coords_from_scalers)
from inr_isr.prediction import predict_log10_on_coords as _predict_log10_on_coords
from inr_isr.analysis import (
    predict_log10_and_gradients_on_coords as _predict_log10_and_gradients_on_coords,
    compute_error_metrics as _compute_error_metrics,
    select_analyzer_time_indices as _select_analyzer_time_indices,
    make_analyzer_dense_grid as _make_analyzer_dense_grid,
)
from models import MLPINR
from synthetic_plasma import (
    MovingGaussianPatch,
    evaluate_synthetic_plasma,
    evaluate_integration_averaged_plasma,
)


# ============================================================
# USER SETTINGS: EDIT THESE FOR EACH ANALYSIS RUN
# ============================================================

# Folder produced by synthetic_train_3d.py
RUN_DIR = Path("outputs/synthetic_train_high_amp_win0_diag_1500")

# Synthetic case used to train that model.
SYNTHETIC_CSV = Path("outputs/synthetic_high_amp_left_right/synthetic_observations.csv")
SYNTHETIC_CONFIG = Path("outputs/synthetic_high_amp_left_right/synthetic_config.json")

# Checkpoint inside RUN_DIR. Usually model_final.pt is fine.
CHECKPOINT_NAME = "model_final.pt"

# Output folder inside RUN_DIR.
ANALYSIS_SUBDIR = "error_analysis"

# Dense evaluation grid. This is independent from the training collocation grid.
GRID_NX = 250
GRID_NY = 250

# Domain mode:
#   "full_domain"      -> use [-domain_size/2, +domain_size/2] from synthetic_config.json.
#                         This shows interpolation/extrapolation over the whole synthetic box.
#   "training_extent"  -> use the x/y min/max stored in the model coordinate scalers.
#                         This avoids evaluating outside the observed coordinate range.
DOMAIN_MODE = "full_domain"

# Time selection:
#   "first_middle_last" -> analyze first, middle, and last available time records.
#   "all"               -> analyze all available time records.
#   list[int]            -> explicit time_index list, e.g. [0, 10, 20, 30].
TIME_SELECTION: str | list[int] = "first_middle_last"

# Model evaluation chunk size. Lower if GPU memory is tight.
PREDICT_CHUNK_SIZE = 65536
GRADIENT_CHUNK_SIZE = 16384

# Compute model first derivatives and compare against analytical truth.
# This is more expensive than density-only error, but useful for INR validation.
COMPUTE_GRADIENT_ERRORS = True

# Save dense CSV files for each analyzed time.
SAVE_DENSE_CSV = True

# Plotting.
CMAP_FIELD = "plasma"
CMAP_ERROR = "RdBu_r"
DPI = 200

# Device. "auto" requires CUDA; set "cpu" for the explicit CPU alternative.
DEVICE = "auto"


@dataclass(frozen=True)
class ReconstructionAnalysisConfig:
    """All inputs and numerical controls for one dense synthetic reconstruction."""

    run_dir: Path
    synthetic_csv: Path
    synthetic_config: Path
    checkpoint_name: str = "model_final.pt"
    analysis_subdir: str = "error_analysis"
    grid_nx: int = 250
    grid_ny: int = 250
    domain_mode: str = "full_domain"
    time_selection: str | list[int] = "first_middle_last"
    predict_chunk_size: int = 65_536
    gradient_chunk_size: int = 16_384
    compute_gradient_errors: bool = True
    save_dense_csv: bool = True
    device: str = "auto"


def config_from_module_settings() -> ReconstructionAnalysisConfig:
    """Build the legacy standalone configuration from the module-level settings."""

    return ReconstructionAnalysisConfig(
        run_dir=Path(RUN_DIR),
        synthetic_csv=Path(SYNTHETIC_CSV),
        synthetic_config=Path(SYNTHETIC_CONFIG),
        checkpoint_name=CHECKPOINT_NAME,
        analysis_subdir=ANALYSIS_SUBDIR,
        grid_nx=GRID_NX,
        grid_ny=GRID_NY,
        domain_mode=DOMAIN_MODE,
        time_selection=TIME_SELECTION,
        predict_chunk_size=PREDICT_CHUNK_SIZE,
        gradient_chunk_size=GRADIENT_CHUNK_SIZE,
        compute_gradient_errors=COMPUTE_GRADIENT_ERRORS,
        save_dense_csv=SAVE_DENSE_CSV,
        device=DEVICE,
    )


# ============================================================
# SYNTHETIC TRUTH EVALUATION
# ============================================================


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    """Load one UTF-8 JSON object."""

    with open(path, "r") as f:
        return json.load(f)


def torch_load_checkpoint(path: Path, device: torch.device):
    """Retained legacy public adapter; shared implementation preserves xyt policy.

    See inr_isr checkpoints/transforms/prediction for shapes, units, errors, RNG
    and side effects. Gradient/truth/plot analysis remains the native recipe.
    """
    return _torch_load_checkpoint(path, device)


def choose_device(device_name: str = DEVICE) -> torch.device:
    """Resolve the shared CUDA-required default or explicit CPU alternative."""
    from inr_isr.device import resolve_device
    return resolve_device(device_name)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """Retained legacy public adapter; shared implementation preserves xyt policy.

    See inr_isr checkpoints/transforms/prediction for shapes, units, errors, RNG
    and side effects. Gradient/truth/plot analysis remains the native recipe.
    """
    return _build_model_from_checkpoint(checkpoint, device)


def patches_from_config(config: dict) -> list[MovingGaussianPatch]:
    """Reconstruct synthetic morphology objects from the saved case config."""

    patches_cfg = config.get("patches")
    if not patches_cfg:
        raise KeyError("synthetic_config.json is missing the 'patches' list.")

    patches = []
    for p in patches_cfg:
        patches.append(
            MovingGaussianPatch(
                name=str(p.get("name", "patch")),
                amplitude_m3=float(p["amplitude_m3"]),
                sigma_x_km=float(p["sigma_x_km"]),
                sigma_y_km=float(p["sigma_y_km"]),
                x0_km=float(p["x0_km"]),
                y0_km=float(p["y0_km"]),
                vx_km_s=float(p["vx_km_s"]),
                vy_km_s=float(p["vy_km_s"]),
            )
        )
    return patches


def denormalize_target(pred_norm: np.ndarray, target_scaler: dict):
    """Retained legacy public adapter; shared implementation preserves xyt policy.

    See inr_isr checkpoints/transforms/prediction for shapes, units, errors, RNG
    and side effects. Gradient/truth/plot analysis remains the native recipe.
    """
    return _denormalize_target(pred_norm, target_scaler)


def normalize_coords_from_scalers(x_km: np.ndarray, y_km: np.ndarray, t_sec: np.ndarray, coord_scalers: dict):
    """Retained legacy public adapter; shared implementation preserves xyt policy.

    See inr_isr checkpoints/transforms/prediction for shapes, units, errors, RNG
    and side effects. Gradient/truth/plot analysis remains the native recipe.
    """
    return _normalize_coords_from_scalers(x_km, y_km, t_sec, coord_scalers)


def predict_log10_on_coords(model: torch.nn.Module, coords_norm: np.ndarray, target_scaler: dict, device: torch.device, chunk_size: int):
    """Retained legacy public adapter; shared implementation preserves xyt policy.

    See inr_isr checkpoints/transforms/prediction for shapes, units, errors, RNG
    and side effects. Gradient/truth/plot analysis remains the native recipe.
    """
    return _predict_log10_on_coords(model, coords_norm, target_scaler, device, chunk_size)


def predict_log10_and_gradients_on_coords(
    model: torch.nn.Module,
    coords_norm: np.ndarray,
    coord_scalers: dict,
    target_scaler: dict,
    device: torch.device,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Keep the public xyt gradient API; the shared chain rule owns the work."""
    return _predict_log10_and_gradients_on_coords(
        model, coords_norm, coord_scalers, target_scaler, device, chunk_size)


def compute_metrics(error: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Summarize one error vector with stable prefixed column names."""
    return _compute_error_metrics(error, prefix)


def select_time_indices(
    obs_df: pd.DataFrame,
    selection: str | list[int] = TIME_SELECTION,
) -> list[int]:
    """Resolve explicit or named time selection against available records."""
    return _select_analyzer_time_indices(obs_df, selection, "time selection")


def make_dense_grid(
    config: dict,
    coord_scalers: dict,
    grid_nx: int = GRID_NX,
    grid_ny: int = GRID_NY,
    domain_mode: str = DOMAIN_MODE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the configured full-domain or training-extent mesh."""
    return _make_analyzer_dense_grid(
        config, coord_scalers, grid_nx, grid_ny, domain_mode, "domain mode")


def plot_truth_pred_error(
    df_time: pd.DataFrame,
    obs_time: pd.DataFrame,
    time_index: int,
    out_path: Path,
) -> None:
    """Save truth, prediction, and signed physical-density error maps."""

    x_unique = np.sort(df_time["x_km"].unique())
    y_unique = np.sort(df_time["y_km"].unique())

    nx = len(x_unique)
    ny = len(y_unique)

    X = df_time["x_km"].to_numpy().reshape(ny, nx)
    Y = df_time["y_km"].to_numpy().reshape(ny, nx)
    truth = df_time["true_log10_Ne"].to_numpy().reshape(ny, nx)
    pred = df_time["pred_log10_Ne"].to_numpy().reshape(ny, nx)

    # Error map shown in the third panel:
    # compute the difference in physical density units first, then compress
    # the dynamic range with a signed log transform.  This avoids treating
    # pred_log10 - true_log10 as an additive density error.
    err = df_time["signed_log10_abs_error_Ne"].to_numpy().reshape(ny, nx)

    vmin = float(min(np.nanmin(truth), np.nanmin(pred)))
    vmax = float(max(np.nanmax(truth), np.nanmax(pred)))
    err_abs = float(np.nanmax(np.abs(err)))
    if err_abs <= 0 or not np.isfinite(err_abs):
        err_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, truth, shading="auto", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
    axes[0].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[0].set_title("Truth log10(Ne)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, pred, shading="auto", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
    axes[1].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[1].set_title("INR prediction log10(Ne)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, err, shading="auto", cmap=CMAP_ERROR, vmin=-err_abs, vmax=err_abs)
    axes[2].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[2].set_title("Signed log10(|pred_Ne - true_Ne| + 1)")
    fig.colorbar(im2, ax=axes[2])

    t_min = float(df_time["t_min"].median())
    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Synthetic reconstruction | time_index={time_index} | t={t_min:.1f} min")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_error(
    df_time: pd.DataFrame,
    obs_time: pd.DataFrame,
    time_index: int,
    out_path: Path,
) -> None:
    """Save truth, predicted, and residual horizontal-gradient magnitudes."""

    x_unique = np.sort(df_time["x_km"].unique())
    y_unique = np.sort(df_time["y_km"].unique())
    nx = len(x_unique)
    ny = len(y_unique)

    X = df_time["x_km"].to_numpy().reshape(ny, nx)
    Y = df_time["y_km"].to_numpy().reshape(ny, nx)

    err_grad_mag = df_time["error_grad_xy_mag"].to_numpy().reshape(ny, nx)
    pred_grad_mag = df_time["pred_grad_xy_mag"].to_numpy().reshape(ny, nx)
    true_grad_mag = df_time["true_grad_xy_mag"].to_numpy().reshape(ny, nx)

    vmax_grad = float(max(np.nanmax(pred_grad_mag), np.nanmax(true_grad_mag)))
    vmax_err = float(np.nanmax(err_grad_mag))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, true_grad_mag, shading="auto", cmap=CMAP_FIELD, vmin=0.0, vmax=vmax_grad)
    axes[0].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[0].set_title("Truth |grad_xy log10Ne|")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, pred_grad_mag, shading="auto", cmap=CMAP_FIELD, vmin=0.0, vmax=vmax_grad)
    axes[1].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[1].set_title("INR |grad_xy log10Ne|")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, err_grad_mag, shading="auto", cmap=CMAP_FIELD, vmin=0.0, vmax=vmax_err)
    axes[2].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[2].set_title("|pred grad_xy - truth grad_xy|")
    fig.colorbar(im2, ax=axes[2])

    t_min = float(df_time["t_min"].median())
    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Synthetic gradient error | time_index={time_index} | t={t_min:.1f} min")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_reconstruction(settings: ReconstructionAnalysisConfig) -> None:
    """Evaluate one trained SIREN against the matching integration-averaged truth."""

    run_dir = Path(settings.run_dir)
    synthetic_csv = Path(settings.synthetic_csv)
    synthetic_config = Path(settings.synthetic_config)
    checkpoint_path = run_dir / settings.checkpoint_name
    out_dir = run_dir / settings.analysis_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not synthetic_csv.exists():
        raise FileNotFoundError(f"Synthetic CSV not found: {synthetic_csv}")
    if not synthetic_config.exists():
        raise FileNotFoundError(f"Synthetic config not found: {synthetic_config}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device(settings.device)
    print("Synthetic reconstruction analysis")
    print(f"  run_dir:          {run_dir}")
    print(f"  checkpoint:       {checkpoint_path}")
    print(f"  synthetic_csv:    {synthetic_csv}")
    print(f"  synthetic_config: {synthetic_config}")
    print(f"  output_dir:       {out_dir}")
    print(f"  device:           {device}")
    print(f"  grid:             {settings.grid_nx} x {settings.grid_ny}")
    print(f"  domain mode:      {settings.domain_mode}")
    print(f"  gradients:        {settings.compute_gradient_errors}")
    print()

    obs_df = pd.read_csv(synthetic_csv)
    config = load_json(synthetic_config)
    patches = patches_from_config(config)

    checkpoint = torch_load_checkpoint(checkpoint_path, device=device)
    model = build_model_from_checkpoint(checkpoint, device=device)

    coord_scalers = checkpoint["coord_scalers"]
    target_scaler = checkpoint["target_scaler"]

    time_indices = select_time_indices(obs_df, settings.time_selection)
    print(f"Time indices selected: {time_indices}")

    X, Y = make_dense_grid(
        config=config,
        coord_scalers=coord_scalers,
        grid_nx=settings.grid_nx,
        grid_ny=settings.grid_ny,
        domain_mode=settings.domain_mode,
    )

    summary_rows = []

    for time_index in time_indices:
        obs_time = obs_df[obs_df["time_index"] == time_index].copy()
        if len(obs_time) == 0:
            raise ValueError(f"No observations for time_index={time_index}")

        t_sec = float(obs_time["t_sec"].median())
        t_min = t_sec / 60.0
        T = np.full_like(X, t_sec, dtype=np.float64)

        midpoint_truth = evaluate_synthetic_plasma(
            x_km=X,
            y_km=Y,
            t_sec=T,
            background_ne_m3=float(config["background_ne_m3"]),
            patches=patches,
        )
        integration_time_sec = float(config.get("integration_time_sec", 0.0))
        integration_samples = int(config.get("integration_samples", 1))
        truth = evaluate_integration_averaged_plasma(
            x_km=X,
            y_km=Y,
            t_sec=T,
            integration_time_sec=integration_time_sec,
            integration_samples=integration_samples,
            background_ne_m3=float(config["background_ne_m3"]),
            patches=patches,
        )

        coords_norm = normalize_coords_from_scalers(
            x_km=X.ravel(),
            y_km=Y.ravel(),
            t_sec=T.ravel(),
            coord_scalers=coord_scalers,
        )

        if settings.compute_gradient_errors:
            pred = predict_log10_and_gradients_on_coords(
                model=model,
                coords_norm=coords_norm,
                coord_scalers=coord_scalers,
                target_scaler=target_scaler,
                device=device,
                chunk_size=settings.gradient_chunk_size,
            )
            pred_log10 = pred["pred_log10_Ne"]
        else:
            pred_log10 = predict_log10_on_coords(
                model=model,
                coords_norm=coords_norm,
                target_scaler=target_scaler,
                device=device,
                chunk_size=settings.predict_chunk_size,
            )
            pred = {"pred_log10_Ne": pred_log10}

        true_log10 = truth["log10_Ne"].ravel()

        # The network predicts log10(Ne), because that is the training target.
        # There are two different, useful errors:
        #
        #   1. error_log10_Ne = pred_log10 - true_log10
        #      This is a dex/log-ratio error: log10(pred_Ne / true_Ne).
        #      It is useful, but it is NOT an additive physical density error.
        #
        #   2. error_Ne = pred_Ne - true_Ne
        #      This is the physical electron-density error in m^-3.
        #      For plotting, signed_log10_abs_error_Ne compresses this physical
        #      error after the linear-space subtraction.
        true_Ne = truth["Ne"].ravel()
        pred_Ne = np.power(10.0, pred_log10)

        error_log10_Ne = pred_log10 - true_log10
        abs_error_log10_Ne = np.abs(error_log10_Ne)

        error_Ne = pred_Ne - true_Ne
        abs_error_Ne = np.abs(error_Ne)
        rel_error_Ne = error_Ne / np.maximum(true_Ne, 1.0e-30)
        abs_rel_error_Ne = np.abs(rel_error_Ne)

        # Log-compressed physical error for signed error maps.
        # The +1 avoids log10(0). The sign preserves over/under prediction.
        signed_log10_abs_error_Ne = np.sign(error_Ne) * np.log10(abs_error_Ne + 1.0)

        midpoint_true_Ne = midpoint_truth["Ne"].ravel()
        midpoint_true_log10 = midpoint_truth["log10_Ne"].ravel()
        midpoint_error_Ne = pred_Ne - midpoint_true_Ne
        midpoint_error_log10_Ne = pred_log10 - midpoint_true_log10

        dense_df = pd.DataFrame(
            {
                "time_index": np.full(X.size, int(time_index), dtype=int),
                "t_sec": np.full(X.size, t_sec, dtype=float),
                "t_min": np.full(X.size, t_min, dtype=float),
                "integration_time_sec": np.full(X.size, integration_time_sec, dtype=float),
                "x_km": X.ravel(),
                "y_km": Y.ravel(),
                "true_Ne": true_Ne,
                "pred_Ne": pred_Ne,
                "error_Ne": error_Ne,
                "abs_error_Ne": abs_error_Ne,
                "rel_error_Ne": rel_error_Ne,
                "abs_rel_error_Ne": abs_rel_error_Ne,
                "signed_log10_abs_error_Ne": signed_log10_abs_error_Ne,
                "true_log10_Ne": true_log10,
                "pred_log10_Ne": pred_log10,
                "error_log10_Ne": error_log10_Ne,
                "abs_error_log10_Ne": abs_error_log10_Ne,
                "midpoint_true_Ne": midpoint_true_Ne,
                "midpoint_true_log10_Ne": midpoint_true_log10,
                "midpoint_error_Ne": midpoint_error_Ne,
                "midpoint_abs_error_Ne": np.abs(midpoint_error_Ne),
                "midpoint_error_log10_Ne": midpoint_error_log10_Ne,
                "true_dlog10Ne_dx_km": truth["true_dlog10Ne_dx_km"].ravel(),
                "true_dlog10Ne_dy_km": truth["true_dlog10Ne_dy_km"].ravel(),
                "true_dlog10Ne_dt_sec": truth["true_dlog10Ne_dt_sec"].ravel(),
            }
        )

        row = {
            "time_index": int(time_index),
            "t_sec": t_sec,
            "t_min": t_min,
            "n_grid_points": int(X.size),
            "domain_mode": settings.domain_mode,
            "grid_nx": int(settings.grid_nx),
            "grid_ny": int(settings.grid_ny),
            "integration_time_sec": integration_time_sec,
        }
        # Keep the original log-space/dex metrics, but also add physical
        # linear-density metrics.
        row.update(compute_metrics(error_log10_Ne, prefix="log10_"))
        row.update(compute_metrics(error_Ne, prefix="Ne_"))
        row.update(compute_metrics(rel_error_Ne, prefix="rel_Ne_"))
        row.update(compute_metrics(signed_log10_abs_error_Ne, prefix="signed_log10_abs_Ne_"))
        row.update(compute_metrics(midpoint_error_Ne, prefix="midpoint_Ne_"))
        row.update(compute_metrics(midpoint_error_log10_Ne, prefix="midpoint_log10_"))

        if settings.compute_gradient_errors:
            dense_df["pred_dlog10Ne_dx_km"] = pred["pred_dlog10Ne_dx_km"]
            dense_df["pred_dlog10Ne_dy_km"] = pred["pred_dlog10Ne_dy_km"]
            dense_df["pred_dlog10Ne_dt_sec"] = pred["pred_dlog10Ne_dt_sec"]

            dense_df["error_dlog10Ne_dx_km"] = dense_df["pred_dlog10Ne_dx_km"] - dense_df["true_dlog10Ne_dx_km"]
            dense_df["error_dlog10Ne_dy_km"] = dense_df["pred_dlog10Ne_dy_km"] - dense_df["true_dlog10Ne_dy_km"]
            dense_df["error_dlog10Ne_dt_sec"] = dense_df["pred_dlog10Ne_dt_sec"] - dense_df["true_dlog10Ne_dt_sec"]

            dense_df["true_grad_xy_mag"] = np.sqrt(
                dense_df["true_dlog10Ne_dx_km"] ** 2
                + dense_df["true_dlog10Ne_dy_km"] ** 2
            )
            dense_df["pred_grad_xy_mag"] = np.sqrt(
                dense_df["pred_dlog10Ne_dx_km"] ** 2
                + dense_df["pred_dlog10Ne_dy_km"] ** 2
            )
            dense_df["error_grad_xy_mag"] = np.sqrt(
                dense_df["error_dlog10Ne_dx_km"] ** 2
                + dense_df["error_dlog10Ne_dy_km"] ** 2
            )

            row.update(compute_metrics(dense_df["error_dlog10Ne_dx_km"].to_numpy(), prefix="grad_x_"))
            row.update(compute_metrics(dense_df["error_dlog10Ne_dy_km"].to_numpy(), prefix="grad_y_"))
            row.update(compute_metrics(dense_df["error_dlog10Ne_dt_sec"].to_numpy(), prefix="grad_t_"))
            row.update(compute_metrics(dense_df["error_grad_xy_mag"].to_numpy(), prefix="grad_xy_mag_"))

        summary_rows.append(row)

        plot_truth_pred_error(
            df_time=dense_df,
            obs_time=obs_time,
            time_index=int(time_index),
            out_path=out_dir / f"truth_pred_error_time_{int(time_index):04d}.png",
        )

        if settings.compute_gradient_errors:
            plot_gradient_error(
                df_time=dense_df,
                obs_time=obs_time,
                time_index=int(time_index),
                out_path=out_dir / f"gradient_error_time_{int(time_index):04d}.png",
            )

        if settings.save_dense_csv:
            dense_path = out_dir / f"dense_reconstruction_time_{int(time_index):04d}.csv"
            dense_df.to_csv(dense_path, index=False)

        print(
            f"time_index={time_index:04d} | "
            f"log10 RMSE={row['log10_rmse']:.4e} | "
            f"Ne RMSE={row['Ne_rmse']:.4e} m^-3 | "
            f"midpoint Ne RMSE={row['midpoint_Ne_rmse']:.4e} m^-3 | "
            f"rel RMSE={row['rel_Ne_rmse']:.4e} | "
            f"signed-log-phys p95={row['signed_log10_abs_Ne_p95_abs']:.4e}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "error_summary_by_time.csv"
    summary_df.to_csv(summary_path, index=False)

    # Also save an all-time average row for quick reference.
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    mean_row = summary_df[numeric_cols].mean().to_dict()
    mean_row["time_index"] = -1
    mean_row["t_sec"] = np.nan
    mean_row["t_min"] = np.nan
    mean_row["label"] = "mean_over_selected_times"
    mean_path = out_dir / "error_summary_mean.csv"
    pd.DataFrame([mean_row]).to_csv(mean_path, index=False)

    print()
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {mean_path}")
    print(f"  plots/CSVs in {out_dir}")
    print("DONE")


def main() -> None:
    """Run the standalone module-level analysis configuration."""

    analyze_reconstruction(config_from_module_settings())


if __name__ == "__main__":
    main()
