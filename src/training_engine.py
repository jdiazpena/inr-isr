# -*- coding: utf-8 -*-
"""Shared optimization engine for synthetic and real-radar x-y-time SIRENs.

The engine owns the common training sequence and equations. Entry-point modules own
data construction and presentation-specific plots. Keeping the sequence centralized
prevents synthetic and radar training behavior from drifting apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import MLPINR
from training_common import (
    append_csv_row,
    clamp_float,
    compute_metrics,
    curvature_losses_xy_t,
    derivative_diagnostics_xy_t,
    make_collocation_pool,
    parameter_grad_norm_from_existing_grads,
    parameter_grad_norm_from_loss,
    ramp_weight,
    safe_ratio,
    sample_batch,
    sample_collocation_points,
    select_plot_time_indices,
    set_seed,
    update_ema_scalar,
)

def train_window(
    args: argparse.Namespace,
    *,
    dataset_factory: Callable[[argparse.Namespace], Any],
    plot_history_fn: Callable[[Path, Path], None],
    plot_diagnostics_fn: Callable[[Path, Path], None],
    plot_prediction_fn: Callable[..., None],
    diagnostics_without_regularization: bool,
) -> None:
    """Train one x-y-time SIREN window without changing experiment semantics.

    The measured-point objective is MSE in normalized log10 electron density. The
    optional spatial and temporal terms penalize second derivatives at collocation
    coordinates. They are soft smoothness priors, not equations of plasma motion.
    Reference-ratio mode adapts each lambda against a stabilized data-loss reference.

    `dataset_factory` is the only data-source boundary: synthetic CSV and AMISR HDF5
    inputs therefore use the same SIREN, optimizer, loss equations, checkpointing,
    and diagnostics.
    """

    # This call must remain before collocation sampling and model construction. Their
    # random-number order is part of reproducible training behavior.
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"

    if history_path.exists() and not args.resume_history:
        history_path.unlink()

    # Save the effective flat CLI/config values needed to reproduce this exact run.
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
    dataset = dataset_factory(args)

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
    # A positive target ratio activates the corresponding prior in adaptive mode;
    # fixed-lambda mode activates it only when the fixed weight is positive.
    if args.reference_loss_weights:
        use_xy_curv = args.target_xy_ratio > 0.0
        use_t_curv = args.target_t_ratio > 0.0
    else:
        use_xy_curv = args.lambda_curv_xy > 0.0
        use_t_curv = args.lambda_curv_t > 0.0

    if diagnostics_without_regularization:
        diagnostic_use_xy = args.num_diagnostic_collocation != 0
        diagnostic_use_t = args.num_diagnostic_collocation != 0
    else:
        diagnostic_use_xy = use_xy_curv
        diagnostic_use_t = use_t_curv
    needs_collocation_pool = (
        use_xy_curv or use_t_curv or diagnostic_use_xy or diagnostic_use_t
    )

    if needs_collocation_pool:
        collocation_pool, collocation_radius_km, collocation_valid_fraction, collocation_n_times = make_collocation_pool(
            dataset=dataset,
            df=df,
            grid_nx=args.collocation_grid_nx,
            grid_ny=args.collocation_grid_ny,
            padding_frac=args.grid_padding_frac,
            nearest_radius_factor=args.nearest_radius_factor,
        )

        collocation_pool = collocation_pool.to(device)

        diagnostic_collocation_points = sample_collocation_points(
            collocation_pool=collocation_pool,
            num_collocation=args.num_diagnostic_collocation,
        ).detach()

        print()
        print("Collocation points:")
        print(f"  pool size:             {collocation_pool.shape[0]}")
        print(f"  sample per step:       {args.num_collocation}")
        print(f"  diagnostic probe size: {diagnostic_collocation_points.shape[0]}")
        print(f"  time records used:     {collocation_n_times}")
        print(f"  nearest radius [km]:   {collocation_radius_km:.3f}")
        print(f"  valid grid fraction:   {collocation_valid_fraction:.3f}")
        print(f"  curvature loss xy:     {use_xy_curv}")
        print(f"  curvature loss t:      {use_t_curv}")
        print(f"  diagnostic xy:         {diagnostic_use_xy}")
        print(f"  diagnostic t:          {diagnostic_use_t}")
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
        diagnostic_collocation_points = None

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

        # Fixed-probe derivative diagnostics.
        "diag_curv_xy_probe",
        "diag_curv_t_probe",
        "fxx_rms",
        "fxx_meanabs",
        "fxx_maxabs",
        "fxx_frac_near_zero",
        "fxx_frac_exact_zero",
        "fxy_rms",
        "fxy_meanabs",
        "fxy_maxabs",
        "fxy_frac_near_zero",
        "fxy_frac_exact_zero",
        "fyy_rms",
        "fyy_meanabs",
        "fyy_maxabs",
        "fyy_frac_near_zero",
        "fyy_frac_exact_zero",
        "ftt_rms",
        "ftt_meanabs",
        "ftt_maxabs",
        "ftt_frac_near_zero",
        "ftt_frac_exact_zero",

        # Parameter-gradient diagnostics.
        "component_grad_computed",
        "grad_norm_total",
        "grad_norm_data",
        "grad_norm_xy_weighted",
        "grad_norm_t_weighted",

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

    derivative_diag_keys = [
        "diag_curv_xy_probe",
        "diag_curv_t_probe",
        "fxx_rms",
        "fxx_meanabs",
        "fxx_maxabs",
        "fxx_frac_near_zero",
        "fxx_frac_exact_zero",
        "fxy_rms",
        "fxy_meanabs",
        "fxy_maxabs",
        "fxy_frac_near_zero",
        "fxy_frac_exact_zero",
        "fyy_rms",
        "fyy_meanabs",
        "fyy_maxabs",
        "fyy_frac_near_zero",
        "fyy_frac_exact_zero",
        "ftt_rms",
        "ftt_meanabs",
        "ftt_maxabs",
        "ftt_frac_near_zero",
        "ftt_frac_exact_zero",
    ]

    grad_diag_keys = [
        "component_grad_computed",
        "grad_norm_total",
        "grad_norm_data",
        "grad_norm_xy_weighted",
        "grad_norm_t_weighted",
    ]

    latest_derivative_diag = {
        key: float("nan")
        for key in derivative_diag_keys
    }

    latest_grad_diag = {
        "component_grad_computed": False,
        "grad_norm_total": float("nan"),
        "grad_norm_data": float("nan"),
        "grad_norm_xy_weighted": float("nan"),
        "grad_norm_t_weighted": float("nan"),
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

        # --------------------------------------------------------
        # Optional parameter-gradient diagnostics.
        #
        # These answer whether each loss component is still pushing
        # the network weights. They are more expensive than the normal
        # training step, so they are computed only every
        # component_grad_every steps.
        # --------------------------------------------------------
        compute_component_grad_diag = (
            args.component_grad_every > 0
            and (
                step == 1
                or step % args.component_grad_every == 0
                or step == args.num_steps
            )
        )

        latest_grad_diag = {
            "component_grad_computed": bool(compute_component_grad_diag),
            "grad_norm_total": float("nan"),
            "grad_norm_data": float("nan"),
            "grad_norm_xy_weighted": float("nan"),
            "grad_norm_t_weighted": float("nan"),
        }

        trainable_params = [
            param
            for param in model.parameters()
            if param.requires_grad
        ]

        if compute_component_grad_diag:
            latest_grad_diag["grad_norm_data"] = parameter_grad_norm_from_loss(
                loss=data_loss,
                parameters=trainable_params,
                retain_graph=True,
            )

            latest_grad_diag["grad_norm_xy_weighted"] = parameter_grad_norm_from_loss(
                loss=curv_xy_weighted,
                parameters=trainable_params,
                retain_graph=True,
            )

            latest_grad_diag["grad_norm_t_weighted"] = parameter_grad_norm_from_loss(
                loss=curv_t_weighted,
                parameters=trainable_params,
                retain_graph=True,
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        if compute_component_grad_diag:
            latest_grad_diag["grad_norm_total"] = parameter_grad_norm_from_existing_grads(
                parameters=trainable_params,
            )

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

            if diagnostic_collocation_points is not None:
                latest_derivative_diag = derivative_diagnostics_xy_t(
                    model=model,
                    coords_col=diagnostic_collocation_points,
                    use_xy=diagnostic_use_xy,
                    use_t=diagnostic_use_t,
                    zero_eps=args.deriv_zero_epsilon,
                )
            else:
                latest_derivative_diag = {
                    key: float("nan")
                    for key in derivative_diag_keys
                }

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

                "diag_curv_xy_probe": latest_derivative_diag["diag_curv_xy_probe"],
                "diag_curv_t_probe": latest_derivative_diag["diag_curv_t_probe"],
                "fxx_rms": latest_derivative_diag["fxx_rms"],
                "fxx_meanabs": latest_derivative_diag["fxx_meanabs"],
                "fxx_maxabs": latest_derivative_diag["fxx_maxabs"],
                "fxx_frac_near_zero": latest_derivative_diag["fxx_frac_near_zero"],
                "fxx_frac_exact_zero": latest_derivative_diag["fxx_frac_exact_zero"],
                "fxy_rms": latest_derivative_diag["fxy_rms"],
                "fxy_meanabs": latest_derivative_diag["fxy_meanabs"],
                "fxy_maxabs": latest_derivative_diag["fxy_maxabs"],
                "fxy_frac_near_zero": latest_derivative_diag["fxy_frac_near_zero"],
                "fxy_frac_exact_zero": latest_derivative_diag["fxy_frac_exact_zero"],
                "fyy_rms": latest_derivative_diag["fyy_rms"],
                "fyy_meanabs": latest_derivative_diag["fyy_meanabs"],
                "fyy_maxabs": latest_derivative_diag["fyy_maxabs"],
                "fyy_frac_near_zero": latest_derivative_diag["fyy_frac_near_zero"],
                "fyy_frac_exact_zero": latest_derivative_diag["fyy_frac_exact_zero"],
                "ftt_rms": latest_derivative_diag["ftt_rms"],
                "ftt_meanabs": latest_derivative_diag["ftt_meanabs"],
                "ftt_maxabs": latest_derivative_diag["ftt_maxabs"],
                "ftt_frac_near_zero": latest_derivative_diag["ftt_frac_near_zero"],
                "ftt_frac_exact_zero": latest_derivative_diag["ftt_frac_exact_zero"],

                "component_grad_computed": latest_grad_diag["component_grad_computed"],
                "grad_norm_total": latest_grad_diag["grad_norm_total"],
                "grad_norm_data": latest_grad_diag["grad_norm_data"],
                "grad_norm_xy_weighted": latest_grad_diag["grad_norm_xy_weighted"],
                "grad_norm_t_weighted": latest_grad_diag["grad_norm_t_weighted"],

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
        plot_history_fn(history_path, out_dir)
        plot_diagnostics_fn(history_path, out_dir)

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
            plot_prediction_fn(
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
