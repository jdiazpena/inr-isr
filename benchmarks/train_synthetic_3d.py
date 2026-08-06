#-*- coding: utf-8 -*-
"""
synthetic_train_3d.py

Train a windowed 3D INR on synthetic plasma data.

3D here means:
    x, y, time

not:
    x, y, z

Current experiment:
    f_theta(x_norm, y_norm, t_norm) -> normalized log10(Ne)

Loss:
    total_loss =
        data_loss
        + lambda_curv_xy_eff * curv_xy_loss
        + lambda_curv_t_eff  * curv_t_loss

This version can use either fixed lambdas or reference-ratio lambdas.

Fixed-lambda mode:
    lambda_curv_xy_eff and lambda_curv_t_eff are set from command-line
    values, with the usual ramp.

Reference-ratio mode:
    lambdas are adjusted so the weighted curvature terms target a chosen
    fraction of a stable data reference:

        data_reference = max(data_loss_ema, epsilon_data)

        lambda_xy_target = target_xy_ratio * data_reference / curv_xy_raw_ema
        lambda_t_target  = target_t_ratio  * data_reference / curv_t_raw_ema

    The epsilon floor prevents the priors from disappearing when the measured
    radar points become easy to fit.

where:
    data_loss:
        MSE at measured radar points.

    curv_xy_loss:
        spatial curvature penalty:
            mean(fxx^2 + 2 fxy^2 + fyy^2)

    curv_t_loss:
        temporal curvature penalty:
            mean(ftt^2)

Important:
    temporal curvature is NOT temporal gradient.
    It allows linear time evolution but penalizes time wiggles.

This script trains ONE synthetic temporal window.
A later wrapper can loop over many windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from synthetic_dataset import SyntheticPlasmaTimeDataset
from models import MLPINR
from training_config import parse_args_with_optional_json
from training_engine import train_window
from training_common import (
    append_csv_row,
    clamp_float,
    compute_metrics,
    curvature_losses_xy_t,
    derivative_diagnostics_xy_t,
    estimate_nearest_radius,
    evaluate_model_on_coords,
    make_collocation_pool,
    make_query_grid_from_points,
    nearest_distance_mask,
    normalize_xy_t_grid_with_dataset,
    parameter_grad_norm_from_existing_grads,
    parameter_grad_norm_from_loss,
    ramp_weight,
    safe_ratio,
    sample_batch,
    sample_collocation_points,
    select_plot_time_indices,
    set_seed,
    tensor_near_zero_stats,
    update_ema_scalar,
)


# ============================================================
# GENERAL HELPERS
# ============================================================


# ============================================================
# GRID / MASK HELPERS
# ============================================================


# ============================================================
# DERIVATIVE LOSSES
# ============================================================


# ============================================================
# PLOTTING
# ============================================================


def plot_history(
    history_path: Path,
    out_dir: Path,
) -> None:
    hist = pd.read_csv(history_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(hist["step"], hist["total_loss"], marker="o", markersize=3, label="total")
    ax.plot(hist["step"], hist["data_loss"], marker="o", markersize=3, label="data")
    ax.plot(hist["step"], hist["curv_xy_weighted"], marker="o", markersize=3, label="xy curv weighted")
    ax.plot(hist["step"], hist["curv_t_weighted"], marker="o", markersize=3, label="t curv weighted")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Synthetic 3D window INR training loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved history plot: {path}")


def plot_diagnostics(
    history_path: Path,
    out_dir: Path,
) -> None:
    """
    Plot derivative and parameter-gradient diagnostics if present in history.csv.
    """

    hist = pd.read_csv(history_path)

    # ------------------------------------------------------------
    # Derivative RMS diagnostics
    # ------------------------------------------------------------
    deriv_cols = [
        ("fxx_rms", r"$f_{xx}$ RMS"),
        ("fxy_rms", r"$f_{xy}$ RMS"),
        ("fyy_rms", r"$f_{yy}$ RMS"),
        ("ftt_rms", r"$f_{tt}$ RMS"),
    ]

    available_deriv = [
        (col, label)
        for col, label in deriv_cols
        if col in hist.columns
    ]

    if available_deriv:
        fig, ax = plt.subplots(figsize=(8, 5))

        for col, label in available_deriv:
            ax.plot(hist["step"], hist[col], marker="o", markersize=3, label=label)

        ax.set_xlabel("step")
        ax.set_ylabel("derivative RMS")
        ax.set_title("Fixed-probe second-derivative diagnostics")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = out_dir / "derivative_rms_diagnostics.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved derivative RMS diagnostics: {path}")

    # ------------------------------------------------------------
    # Near-zero fractions
    # ------------------------------------------------------------
    zero_cols = [
        ("fxx_frac_near_zero", r"$f_{xx}$ near-zero fraction"),
        ("fxy_frac_near_zero", r"$f_{xy}$ near-zero fraction"),
        ("fyy_frac_near_zero", r"$f_{yy}$ near-zero fraction"),
        ("ftt_frac_near_zero", r"$f_{tt}$ near-zero fraction"),
    ]

    available_zero = [
        (col, label)
        for col, label in zero_cols
        if col in hist.columns
    ]

    if available_zero:
        fig, ax = plt.subplots(figsize=(8, 5))

        for col, label in available_zero:
            ax.plot(hist["step"], hist[col], marker="o", markersize=3, label=label)

        ax.set_xlabel("step")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Fraction of fixed-probe derivatives near zero")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = out_dir / "derivative_near_zero_fraction.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved derivative near-zero diagnostics: {path}")

    # ------------------------------------------------------------
    # Parameter-gradient norms
    # ------------------------------------------------------------
    grad_cols = [
        ("grad_norm_total", "total"),
        ("grad_norm_data", "data"),
        ("grad_norm_xy_weighted", "xy weighted"),
        ("grad_norm_t_weighted", "t weighted"),
    ]

    available_grad = [
        (col, label)
        for col, label in grad_cols
        if col in hist.columns
    ]

    if available_grad:
        fig, ax = plt.subplots(figsize=(8, 5))

        any_plotted = False

        for col, label in available_grad:
            y = pd.to_numeric(hist[col], errors="coerce")

            if y.notna().any():
                ax.plot(hist["step"], y, marker="o", markersize=3, label=label)
                any_plotted = True

        if any_plotted:
            ax.set_xlabel("step")
            ax.set_ylabel("parameter-gradient L2 norm")
            ax.set_title("Loss-component gradient norms")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()

            path = out_dir / "gradient_norm_diagnostics.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved gradient norm diagnostics: {path}")
        else:
            plt.close(fig)


def plot_xy_prediction_at_time(
    model: torch.nn.Module,
    dataset: SyntheticPlasmaTimeDataset,
    df: pd.DataFrame,
    time_index: int,
    out_dir: Path,
    device: torch.device,
    grid_nx: int,
    grid_ny: int,
    grid_padding_frac: float,
    nearest_radius_factor: float,
    grid_chunk_size: int,
    save_grid_csv: bool,
    vmin: float,
    vmax: float,
) -> None:
    df_time = df[df["time_index"] == time_index].copy()

    if len(df_time) == 0:
        raise ValueError(f"No dataframe rows for time_index={time_index}")

    t_sec = float(df_time["t_sec"].median())
    t_min = t_sec / 60.0
    unix_mid = float(df_time["unix_mid"].median()) if "unix_mid" in df_time.columns else np.nan

    X, Y = make_query_grid_from_points(
        x_km=df["x_km"].to_numpy(dtype=float),
        y_km=df["y_km"].to_numpy(dtype=float),
        nx=grid_nx,
        ny=grid_ny,
        padding_frac=grid_padding_frac,
    )

    measured_xy = df_time[["x_km", "y_km"]].to_numpy(dtype=float)

    nearest_radius_km = estimate_nearest_radius(
        measured_xy,
        factor=nearest_radius_factor,
    )

    valid_mask, nearest_dist_grid = nearest_distance_mask(
        X,
        Y,
        measured_xy,
        radius_km=nearest_radius_km,
    )

    coords_grid_np = normalize_xy_t_grid_with_dataset(
        dataset=dataset,
        X=X,
        Y=Y,
        t_sec=t_sec,
    )

    pred_flat = evaluate_model_on_coords(
        model=model,
        coords_np=coords_grid_np,
        dataset=dataset,
        device=device,
        chunk_size=grid_chunk_size,
    )

    pred_grid = pred_flat.reshape(X.shape)

    pred_masked = pred_grid.copy()
    pred_masked[~valid_mask] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        X,
        Y,
        pred_masked,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(im, ax=ax, label="predicted log10(Ne)")

    ax.scatter(
        df_time["x_km"],
        df_time["y_km"],
        c=df_time["log10_Ne"],
        s=35,
        edgecolor="k",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("x east [km]")
    ax.set_ylabel("y north [km]")
    ax.set_title(
        f"Synthetic 3D INR | time_index={time_index} | t={t_min:.1f} min"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig_path = out_dir / f"xy_time_index_{time_index:04d}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved x-y time plot: {fig_path}")

    if save_grid_csv:
        grid_df = pd.DataFrame(
            {
                "time_index": np.full(X.size, int(time_index), dtype=int),
                "unix_mid": np.full(X.size, unix_mid, dtype=float),
                "t_sec": np.full(X.size, t_sec, dtype=float),
                "t_min": np.full(X.size, t_min, dtype=float),
                "x_km": X.ravel(),
                "y_km": Y.ravel(),
                "pred_log10_Ne": pred_grid.ravel(),
                "nearest_dist_km": nearest_dist_grid.ravel(),
                "valid_mask": valid_mask.ravel(),
            }
        )

        csv_path = out_dir / f"grid_prediction_time_index_{time_index:04d}.csv"
        grid_df.to_csv(csv_path, index=False)

        print(f"Saved grid CSV: {csv_path}")


# ============================================================
# TRAINING
# ============================================================


def build_dataset(args: argparse.Namespace) -> SyntheticPlasmaTimeDataset:
    """Load the requested synthetic observation window."""

    return SyntheticPlasmaTimeDataset(
        csv_path=args.synthetic_csv,
        target_col=args.target_col,
        window_start_index=args.window_start_index,
        window_size_records=args.window_size_records,
        verbose=True,
    )


def train(args: argparse.Namespace) -> None:
    """Train one synthetic SIREN window with full derivative-health diagnostics."""

    train_window(
        args,
        dataset_factory=build_dataset,
        plot_history_fn=plot_history,
        plot_diagnostics_fn=plot_diagnostics,
        plot_prediction_fn=plot_xy_prediction_at_time,
        diagnostics_without_regularization=True,
    )


# ============================================================
# ARGUMENTS
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--synthetic_csv",
        type=str,
        default="outputs/synthetic_smoke_test/synthetic_observations.csv",
        help="Synthetic observations CSV generated by synthetic_plasma.py.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="log10_Ne",
        help="Target column in the synthetic CSV.",
    )

    # Window
    parser.add_argument("--window_start_index", type=int, default=0)
    parser.add_argument("--window_size_records", type=int, default=11)

    # Model
    parser.add_argument(
        "--activation",
        type=str,
        default="sine",
        choices=["relu", "tanh", "softplus", "sine"],
    )
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--first_omega_0", type=float, default=5.0)
    parser.add_argument("--hidden_omega_0", type=float, default=5.0)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="0 or >= N means full batch.",
    )
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    # Regularization: fixed-lambda fallback
    parser.add_argument("--lambda_curv_xy", type=float, default=0.0)
    parser.add_argument("--lambda_curv_t", type=float, default=0.0)

    # Regularization: reference-ratio lambda mode
    parser.add_argument(
        "--reference_loss_weights",
        action="store_true",
        help="Use target ratios and data_reference=max(data_loss_ema, epsilon_data) to set lambdas.",
    )
    parser.add_argument("--target_xy_ratio", type=float, default=0.30)
    parser.add_argument("--target_t_ratio", type=float, default=0.30)
    parser.add_argument(
        "--epsilon_data",
        type=float,
        default=1e-6,
        help="Minimum data reference used for lambda calibration.",
    )
    parser.add_argument(
        "--loss_ema_beta",
        type=float,
        default=0.99,
        help="EMA beta for data and raw curvature losses.",
    )
    parser.add_argument(
        "--curvature_ema_floor",
        type=float,
        default=1e-30,
        help="Floor for raw curvature EMA denominator.",
    )
    parser.add_argument(
        "--lambda_smoothing",
        type=float,
        default=0.05,
        help="Fraction of target lambda blended into base lambda at each update.",
    )
    parser.add_argument("--lambda_update_every", type=int, default=10)
    parser.add_argument(
        "--lambda_warmup_steps",
        type=int,
        default=500,
        help="Steps before lambdas are allowed to update in reference mode.",
    )
    parser.add_argument(
        "--freeze_lambdas_after_step",
        type=int,
        default=0,
        help="0 means never freeze. Otherwise freeze lambdas at/after this step.",
    )
    parser.add_argument("--lambda_curv_xy_min", type=float, default=0.0)
    parser.add_argument("--lambda_curv_xy_max", type=float, default=1e-6)
    parser.add_argument("--lambda_curv_t_min", type=float, default=0.0)
    parser.add_argument("--lambda_curv_t_max", type=float, default=1e-6)

    parser.add_argument(
        "--num_collocation",
        type=int,
        default=8192,
        help="0 means use all collocation points.",
    )
    parser.add_argument("--collocation_grid_nx", type=int, default=80)
    parser.add_argument("--collocation_grid_ny", type=int, default=80)
    parser.add_argument("--reg_ramp_frac", type=float, default=0.2)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--summary_every", type=int, default=250)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--resume_history", action="store_true")

    # Diagnostics
    parser.add_argument(
        "--deriv_zero_epsilon",
        type=float,
        default=1e-12,
        help="Threshold for derivative near-zero fractions.",
    )
    parser.add_argument(
        "--num_diagnostic_collocation",
        type=int,
        default=4096,
        help="Fixed collocation probe size for derivative diagnostics.",
    )
    parser.add_argument(
        "--component_grad_every",
        type=int,
        default=500,
        help="Compute expensive per-loss-component parameter gradient norms every N steps. 0 disables.",
    )

    # Grid visualization
    parser.add_argument("--grid_nx", type=int, default=250)
    parser.add_argument("--grid_ny", type=int, default=250)
    parser.add_argument("--grid_padding_frac", type=float, default=0.05)
    parser.add_argument("--grid_chunk_size", type=int, default=65536)
    parser.add_argument("--nearest_radius_factor", type=float, default=2.5)
    parser.add_argument(
        "--num_plot_times",
        type=int,
        default=3,
        help="Number of existing time records to plot. 3 means first/middle/last.",
    )

    # Outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/synthetic_3d_window_reference_reg_diagnostic",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--save_grid_csv", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parse_args_with_optional_json(parser)
    train(args)
