# -*- coding: utf-8 -*-
"""
synthetic_train_3d_window_reference_reg.py

Train a windowed 3D INR on synthetic x-y-time plasma data.

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

This script trains ONE temporal window.
A later wrapper can loop over many windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from synthetic_dataset import SyntheticPlasmaTimeDataset
from models import MLPINR
from training_config import parse_args_with_optional_json
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
        f"Synthetic 3D window INR | time_index={time_index} | t={t_min:.1f} min"
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

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"

    if history_path.exists() and not args.resume_history:
        history_path.unlink()

    config = vars(args).copy()

    config_path = out_dir / "run_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # 1. Load one time window
    # ------------------------------------------------------------
    dataset = SyntheticPlasmaTimeDataset(
        csv_path=args.synthetic_csv,
        target_col=args.target_col,
        window_start_index=args.window_start_index,
        window_size_records=args.window_size_records,
        verbose=True,
    )

    dataset.summary()

    sample = dataset[0]

    full_coords = sample["coords"].to(device)
    full_values = sample["values"].to(device)

    df = dataset.df.copy()

    n_total = full_coords.shape[0]

    print()
    print("Training data:")
    print(f"  measured points: {n_total}")
    print(f"  coords shape:    {tuple(full_coords.shape)}")
    print(f"  values shape:    {tuple(full_values.shape)}")
    print(f"  time records:    {df['time_index'].nunique()}")

    if args.batch_size <= 0 or args.batch_size >= n_total:
        print("  training mode:   full batch")
    else:
        print(f"  training mode:   minibatch, batch_size={args.batch_size}")

    # ------------------------------------------------------------
    # 2. Collocation pool for derivative losses
    # ------------------------------------------------------------
    if args.reference_loss_weights:
        use_xy_curv = args.target_xy_ratio > 0.0
        use_t_curv = args.target_t_ratio > 0.0
    else:
        use_xy_curv = args.lambda_curv_xy > 0.0
        use_t_curv = args.lambda_curv_t > 0.0

    if use_xy_curv or use_t_curv:
        collocation_pool, collocation_radius_km, collocation_valid_fraction, collocation_n_times = make_collocation_pool(
            dataset=dataset,
            df=df,
            grid_nx=args.collocation_grid_nx,
            grid_ny=args.collocation_grid_ny,
            padding_frac=args.grid_padding_frac,
            nearest_radius_factor=args.nearest_radius_factor,
        )

        collocation_pool = collocation_pool.to(device)

        print()
        print("Collocation points:")
        print(f"  pool size:             {collocation_pool.shape[0]}")
        print(f"  sample per step:       {args.num_collocation}")
        print(f"  time records used:     {collocation_n_times}")
        print(f"  nearest radius [km]:   {collocation_radius_km:.3f}")
        print(f"  valid grid fraction:   {collocation_valid_fraction:.3f}")
        print(f"  lambda_curv_xy:        {args.lambda_curv_xy}")
        print(f"  lambda_curv_t:         {args.lambda_curv_t}")
        print(f"  reg_ramp_frac:         {args.reg_ramp_frac}")
        print(f"  reference mode:        {args.reference_loss_weights}")
        if args.reference_loss_weights:
            print(f"  target_xy_ratio:       {args.target_xy_ratio}")
            print(f"  target_t_ratio:        {args.target_t_ratio}")
            print(f"  epsilon_data:          {args.epsilon_data}")
            print(f"  loss_ema_beta:         {args.loss_ema_beta}")
            print(f"  lambda_smoothing:      {args.lambda_smoothing}")
            print(f"  lambda_update_every:   {args.lambda_update_every}")
            print(f"  lambda_warmup_steps:   {args.lambda_warmup_steps}")
            print(f"  freeze_after_step:     {args.freeze_lambdas_after_step}")
    else:
        collocation_pool = None

    # ------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------
    model = MLPINR(
        in_features=dataset.in_features,
        out_features=dataset.out_features,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
        outermost_linear=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print()
    print("Model config:")
    print(f"  in_features:      {dataset.in_features}")
    print(f"  out_features:     {dataset.out_features}")
    print(f"  activation:       {args.activation}")
    print(f"  hidden_features:  {args.hidden_features}")
    print(f"  hidden_layers:    {args.hidden_layers}")
    print(f"  first_omega_0:    {args.first_omega_0}")
    print(f"  hidden_omega_0:   {args.hidden_omega_0}")
    print(f"  lr:               {args.lr}")
    print(f"  num_steps:        {args.num_steps}")

    # ------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------
    history_fields = [
        "step",
        "total_loss",
        "data_loss",
        "curv_xy_raw",
        "curv_xy_weighted",
        "lambda_curv_xy_base",
        "lambda_curv_xy_eff",
        "lambda_curv_xy_target",
        "curv_t_raw",
        "curv_t_weighted",
        "lambda_curv_t_base",
        "lambda_curv_t_eff",
        "lambda_curv_t_target",
        "data_loss_ema",
        "data_reference",
        "curv_xy_raw_ema",
        "curv_t_raw_ema",
        "xy_over_data_inst",
        "t_over_data_inst",
        "xy_over_data_ref",
        "t_over_data_ref",
        "reference_loss_weights",
        "lambda_update_active",
        "lambda_frozen",
        "target_xy_ratio",
        "target_t_ratio",
        "epsilon_data",
        "rmse_log10",
        "mae_log10",
        "bias_log10",
        "max_abs_log10",
        "p95_abs_log10",
        "p99_abs_log10",
    ]

    latest_metrics = {
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "max_abs": np.nan,
        "p95_abs": np.nan,
        "p99_abs": np.nan,
    }
    # ------------------------------------------------------------
    # Best checkpoints after regularization ramp
    # ------------------------------------------------------------
    if args.reg_ramp_frac > 0.0:
        ramp_steps = max(1, int(args.reg_ramp_frac * args.num_steps))
    else:
        ramp_steps = 0

    best_total_after_ramp = float("inf")
    best_data_after_ramp = float("inf")

    best_total_step = None
    best_data_step = None

    best_total_path = out_dir / "model_best_total_after_ramp.pt"
    best_data_path = out_dir / "model_best_data_after_ramp.pt"

    # ------------------------------------------------------------
    # Reference-ratio lambda state
    # ------------------------------------------------------------
    data_loss_ema = None
    curv_xy_raw_ema = None
    curv_t_raw_ema = None

    lambda_curv_xy_base = float(args.lambda_curv_xy)
    lambda_curv_t_base = float(args.lambda_curv_t)

    lambda_curv_xy_target = float(args.lambda_curv_xy)
    lambda_curv_t_target = float(args.lambda_curv_t)

    lambda_frozen = False

    if args.reference_loss_weights:
        lambda_curv_xy_base = 0.0
        lambda_curv_t_base = 0.0
        lambda_curv_xy_target = 0.0
        lambda_curv_t_target = 0.0

    pbar = tqdm(
        range(1, args.num_steps + 1),
        disable=args.disable_tqdm,
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for step in pbar:
        model.train()

        batch_coords, batch_values = sample_batch(
            full_coords,
            full_values,
            args.batch_size,
        )

        pred = model(batch_coords)
        data_loss = F.mse_loss(pred, batch_values)

        if use_xy_curv or use_t_curv:
            coords_col = sample_collocation_points(
                collocation_pool=collocation_pool,
                num_collocation=args.num_collocation,
            )

            curv_xy_raw, curv_t_raw = curvature_losses_xy_t(
                model=model,
                coords_col=coords_col,
                use_xy=use_xy_curv,
                use_t=use_t_curv,
            )
        else:
            curv_xy_raw = data_loss.new_tensor(0.0)
            curv_t_raw = data_loss.new_tensor(0.0)

        # --------------------------------------------------------
        # Update EMA statistics used by reference-ratio weighting.
        # These are diagnostics in fixed-lambda mode and controls in
        # reference mode.
        # --------------------------------------------------------
        data_scalar_current = float(data_loss.detach().item())
        curv_xy_scalar_current = float(curv_xy_raw.detach().item())
        curv_t_scalar_current = float(curv_t_raw.detach().item())

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

        curv_xy_weighted = lambda_curv_xy_eff * curv_xy_raw
        curv_t_weighted = lambda_curv_t_eff * curv_t_raw

        xy_weighted_scalar = float(curv_xy_weighted.detach().item())
        t_weighted_scalar = float(curv_t_weighted.detach().item())

        xy_over_data_inst = safe_ratio(xy_weighted_scalar, data_scalar_current)
        t_over_data_inst = safe_ratio(t_weighted_scalar, data_scalar_current)
        xy_over_data_ref = safe_ratio(xy_weighted_scalar, data_reference)
        t_over_data_ref = safe_ratio(t_weighted_scalar, data_reference)

        # total_loss = data_loss + curv_xy_weighted + curv_t_weighted

        # optimizer.zero_grad(set_to_none=True)
        # total_loss.backward()
        # optimizer.step()

        total_loss = data_loss + curv_xy_weighted + curv_t_weighted

        # --------------------------------------------------------
        # Save best checkpoints after the regularization ramp.
        #
        # This saves the model BEFORE the optimizer step, so the saved
        # weights correspond to the loss values used for the decision.
        # --------------------------------------------------------
        if step > ramp_steps:
            total_scalar = float(total_loss.detach().item())
            data_scalar = float(data_loss.detach().item())

            if total_scalar < best_total_after_ramp:
                best_total_after_ramp = total_scalar
                best_total_step = step

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "coord_scalers": dataset.coord_scalers,
                        "target_scaler": dataset.target_scaler,
                        "checkpoint_type": "best_total_after_ramp",
                        "step": step,
                        "losses": {
                            "total_loss": total_scalar,
                            "data_loss": data_scalar,
                            "curv_xy_raw": float(curv_xy_raw.detach().item()),
                            "curv_xy_weighted": float(curv_xy_weighted.detach().item()),
                            "lambda_curv_xy_eff": float(lambda_curv_xy_eff),
                            "curv_t_raw": float(curv_t_raw.detach().item()),
                            "curv_t_weighted": float(curv_t_weighted.detach().item()),
                            "lambda_curv_t_eff": float(lambda_curv_t_eff),
                            "lambda_curv_xy_base": float(lambda_curv_xy_base),
                            "lambda_curv_t_base": float(lambda_curv_t_base),
                            "lambda_curv_xy_target": float(lambda_curv_xy_target),
                            "lambda_curv_t_target": float(lambda_curv_t_target),
                            "data_loss_ema": float(data_loss_ema),
                            "data_reference": float(data_reference),
                            "curv_xy_raw_ema": float(curv_xy_raw_ema),
                            "curv_t_raw_ema": float(curv_t_raw_ema),
                            "xy_over_data_ref": float(xy_over_data_ref),
                            "t_over_data_ref": float(t_over_data_ref),
                        },
                    },
                    best_total_path,
                )

            if data_scalar < best_data_after_ramp:
                best_data_after_ramp = data_scalar
                best_data_step = step

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "coord_scalers": dataset.coord_scalers,
                        "target_scaler": dataset.target_scaler,
                        "checkpoint_type": "best_data_after_ramp",
                        "step": step,
                        "losses": {
                            "total_loss": total_scalar,
                            "data_loss": data_scalar,
                            "curv_xy_raw": float(curv_xy_raw.detach().item()),
                            "curv_xy_weighted": float(curv_xy_weighted.detach().item()),
                            "lambda_curv_xy_eff": float(lambda_curv_xy_eff),
                            "curv_t_raw": float(curv_t_raw.detach().item()),
                            "curv_t_weighted": float(curv_t_weighted.detach().item()),
                            "lambda_curv_t_eff": float(lambda_curv_t_eff),
                            "lambda_curv_xy_base": float(lambda_curv_xy_base),
                            "lambda_curv_t_base": float(lambda_curv_t_base),
                            "lambda_curv_xy_target": float(lambda_curv_xy_target),
                            "lambda_curv_t_target": float(lambda_curv_t_target),
                            "data_loss_ema": float(data_loss_ema),
                            "data_reference": float(data_reference),
                            "curv_xy_raw_ema": float(curv_xy_raw_ema),
                            "curv_t_raw_ema": float(curv_t_raw_ema),
                            "xy_over_data_ref": float(xy_over_data_ref),
                            "t_over_data_ref": float(t_over_data_ref),
                        },
                    },
                    best_data_path,
                )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % args.summary_every == 0 or step == args.num_steps:
            model.eval()

            with torch.no_grad():
                pred_norm_np = model(full_coords).detach().cpu().numpy()

            pred_df = dataset.make_prediction_dataframe(pred_norm_np)

            metrics = compute_metrics(
                pred=pred_df["pred_log10_Ne"].to_numpy(),
                target=pred_df["log10_Ne"].to_numpy(),
            )

            latest_metrics = metrics

            row = {
                "step": step,
                "total_loss": float(total_loss.item()),
                "data_loss": float(data_loss.item()),
                "curv_xy_raw": float(curv_xy_raw.item()),
                "curv_xy_weighted": float(curv_xy_weighted.item()),
                "lambda_curv_xy_base": float(lambda_curv_xy_base),
                "lambda_curv_xy_eff": float(lambda_curv_xy_eff),
                "lambda_curv_xy_target": float(lambda_curv_xy_target),
                "curv_t_raw": float(curv_t_raw.item()),
                "curv_t_weighted": float(curv_t_weighted.item()),
                "lambda_curv_t_base": float(lambda_curv_t_base),
                "lambda_curv_t_eff": float(lambda_curv_t_eff),
                "lambda_curv_t_target": float(lambda_curv_t_target),
                "data_loss_ema": float(data_loss_ema),
                "data_reference": float(data_reference),
                "curv_xy_raw_ema": float(curv_xy_raw_ema),
                "curv_t_raw_ema": float(curv_t_raw_ema),
                "xy_over_data_inst": float(xy_over_data_inst),
                "t_over_data_inst": float(t_over_data_inst),
                "xy_over_data_ref": float(xy_over_data_ref),
                "t_over_data_ref": float(t_over_data_ref),
                "reference_loss_weights": bool(args.reference_loss_weights),
                "lambda_update_active": bool(lambda_update_active),
                "lambda_frozen": bool(lambda_frozen),
                "target_xy_ratio": float(args.target_xy_ratio),
                "target_t_ratio": float(args.target_t_ratio),
                "epsilon_data": float(args.epsilon_data),
                "rmse_log10": metrics["rmse"],
                "mae_log10": metrics["mae"],
                "bias_log10": metrics["bias"],
                "max_abs_log10": metrics["max_abs"],
                "p95_abs_log10": metrics["p95_abs"],
                "p99_abs_log10": metrics["p99_abs"],
            }

            append_csv_row(history_path, history_fields, row)

        if step == 1 or step % args.log_every == 0 or step == args.num_steps:
            pbar.set_postfix_str(
                f"tot={total_loss.item():.2e} "
                f"data={data_loss.item():.2e} "
                f"ref={data_reference:.1e} "
                f"xyW={curv_xy_weighted.item():.2e} "
                f"tW={curv_t_weighted.item():.2e} "
                f"xyRef={xy_over_data_ref:.2f} "
                f"tRef={t_over_data_ref:.2f} "
                f"lxy={lambda_curv_xy_eff:.1e} "
                f"lt={lambda_curv_t_eff:.1e} "
                f"rmse={latest_metrics['rmse']:.2e}"
            )

    # ------------------------------------------------------------
    # 5. Save model and measured-point predictions
    # ------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        pred_norm_np = model(full_coords).detach().cpu().numpy()

    pred_df = dataset.make_prediction_dataframe(pred_norm_np)

    pred_csv = out_dir / "predictions_at_measured_points.csv"
    pred_df.to_csv(pred_csv, index=False)

    final_metrics = compute_metrics(
        pred=pred_df["pred_log10_Ne"].to_numpy(),
        target=pred_df["log10_Ne"].to_numpy(),
    )

    model_path = out_dir / "model_final.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "coord_scalers": dataset.coord_scalers,
            "target_scaler": dataset.target_scaler,
            "final_metrics": final_metrics,
        },
        model_path,
    )

    if best_total_step is not None:
        print(
            f"Saved best-total checkpoint: {best_total_path} "
            f"(step {best_total_step}, total={best_total_after_ramp:.8e})"
        )

    if best_data_step is not None:
        print(
            f"Saved best-data checkpoint: {best_data_path} "
            f"(step {best_data_step}, data={best_data_after_ramp:.8e})"
        )

    print()
    print("Final measured-point metrics in log10(Ne):")
    for key, value in final_metrics.items():
        print(f"  {key:12s}: {value:.8e}")

    # ------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------
    if not args.no_plots:
        plot_history(history_path, out_dir)

        plot_time_indices = select_plot_time_indices(
            df=df,
            num_plot_times=args.num_plot_times,
        )

        vmin = float(df["log10_Ne"].min())
        vmax = float(df["log10_Ne"].max())

        print()
        print("Plot time indices:")
        print(plot_time_indices)
        print(f"Fixed color scale: vmin={vmin:.6f}, vmax={vmax:.6f}")

        for time_index in plot_time_indices:
            plot_xy_prediction_at_time(
                model=model,
                dataset=dataset,
                df=df,
                time_index=time_index,
                out_dir=out_dir,
                device=device,
                grid_nx=args.grid_nx,
                grid_ny=args.grid_ny,
                grid_padding_frac=args.grid_padding_frac,
                nearest_radius_factor=args.nearest_radius_factor,
                grid_chunk_size=args.grid_chunk_size,
                save_grid_csv=args.save_grid_csv,
                vmin=vmin,
                vmax=vmax,
            )

    print()
    print("DONE")


# ============================================================
# ARGUMENTS
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--synthetic_csv",
        type=str,
        default="outputs/synthetic_high_amp_test/synthetic_observations.csv",
        help="Synthetic observation CSV produced by synthetic_plasma.py.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="log10_Ne",
        help="Target column to train on. Usually log10_Ne.",
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
        default="outputs/radar_3d_window_reference_reg",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--save_grid_csv", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parse_args_with_optional_json(parser)
    train(args)
