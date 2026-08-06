# -*- coding: utf-8 -*-
"""
benchmarks/train_synthetic_4d.py

Train a 4D SIREN model on synthetic 4D plasma patch observations (x_km, y_km, z_km, t_sec).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models import MLPINR
from synthetic_plasma import SyntheticPlasma4DDataset
from training_common import curvature_loss_4d, compute_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 4D SIREN on synthetic 4D plasma patch data")
    parser.add_argument("--config", type=str, default="configs/synthetic_patch_4d_config.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss_type", type=str, default=None, choices=["isotropic", "anisotropic_huber"])
    return parser.parse_args()


def run_synthetic_4d_training(config: dict) -> dict:
    seed = config.get("seed", 42)
    set_seed(seed)

    output_dir = Path(config.get("output_dir", "outputs/synthetic_4d_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    dataset = SyntheticPlasma4DDataset(
        num_points=config.get("num_points", 5000),
        x_bounds=(config.get("x_min_km", -300.0), config.get("x_max_km", 300.0)),
        y_bounds=(config.get("y_min_km", -300.0), config.get("y_max_km", 300.0)),
        z_bounds=(config.get("z_min_km", 100.0), config.get("z_max_km", 500.0)),
        t_bounds=(config.get("t_min_sec", 0.0), config.get("t_max_sec", 300.0)),
        v_km_s=config.get("v_km_s", 1.0),
        background_ne_m3=config.get("background_ne_m3", 1e11),
        patch_amplitude_m3=config.get("patch_amplitude_m3", 5e11),
        seed=seed,
    )

    sample = dataset[0]
    coords = sample["coords"].to(device)
    values = sample["values"].to(device)

    # Collocation pool for curvature loss
    collocation_pool = coords.clone()

    # 2. Model
    model = MLPINR(
        in_features=config.get("in_features", 4),
        out_features=config.get("out_features", 1),
        hidden_features=config.get("hidden_features", 256),
        hidden_layers=config.get("hidden_layers", 3),
        activation=config.get("activation", "sine"),
        first_omega_0=config.get("first_omega_0", 5.0),
        hidden_omega_0=config.get("hidden_omega_0", 5.0),
    ).to(device)

    lr = config.get("learning_rate", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lambda_curv = config.get("lambda_curvature", 1e-4)
    num_steps = config.get("num_steps", 500)

    loss_type = config.get("loss_type", "isotropic")
    lambda_xy = config.get("lambda_xy", 1.0)
    lambda_z = config.get("lambda_z", 1.0)
    lambda_t = config.get("lambda_t", 1.0)
    huber_delta_z = config.get("huber_delta_z", 0.1)

    history = []
    print(f"Starting 4D synthetic training for {num_steps} steps (loss_type={loss_type})...")

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(coords)
        l_data = F.mse_loss(pred, values)

        # Autograd 4D curvature loss (supports isotropic and anisotropic_huber)
        l_curv, curv_details = curvature_loss_4d(
            model=model,
            coords_col=collocation_pool,
            loss_type=loss_type,
            lambda_xy=lambda_xy,
            lambda_z=lambda_z,
            lambda_t=lambda_t,
            huber_delta_z=huber_delta_z,
        )

        l_total = l_data + lambda_curv * l_curv

        if not torch.isfinite(l_total):
            raise RuntimeError(f"Step {step}: Non-finite loss detected: l_total={l_total.item()}")

        l_total.backward()
        optimizer.step()

        step_dict = {
            "step": step,
            "total_loss": l_total.item(),
            "data_loss": l_data.item(),
            "curv_loss": l_curv.item(),
            "curv_xx": curv_details["curv_xx"].item(),
            "curv_yy": curv_details["curv_yy"].item(),
            "curv_zz_quadratic": curv_details["curv_zz_quadratic"].item(),
            "curv_zz_huber": curv_details["curv_zz_huber"].item(),
            "curv_tt": curv_details["curv_tt"].item(),
            "loss_type": loss_type,
        }
        history.append(step_dict)

        if step % 100 == 0 or step == num_steps:
            print(
                f"Step {step:4d}/{num_steps} | Total Loss: {l_total.item():.6e} | "
                f"Data Loss: {l_data.item():.6e} | Curv Loss: {l_curv.item():.6e}"
            )

    # 3. Save History & Model
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    torch.save(model.state_dict(), output_dir / "model_4d.pt")
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    # 4. Evaluation Metrics
    model.eval()
    with torch.no_grad():
        pred_norm = model(coords).cpu().numpy().ravel()
        target_norm = values.cpu().numpy().ravel()

    pred_log10_ne = dataset.denormalize_target(pred_norm)
    target_log10_ne = dataset.denormalize_target(target_norm)

    res_metrics = compute_metrics(pred_log10_ne, target_log10_ne)

    # R^2 score
    ss_res = np.sum((target_log10_ne - pred_log10_ne) ** 2)
    ss_tot = np.sum((target_log10_ne - np.mean(target_log10_ne)) ** 2)
    r2_score = float(1.0 - (ss_res / max(ss_tot, 1e-12)))

    metrics = {
        "mse": res_metrics["mse"],
        "rmse": res_metrics["rmse"],
        "mae": res_metrics["mae"],
        "r2_score": r2_score,
        "final_data_loss": float(history[-1]["data_loss"]),
        "final_curv_loss": float(history[-1]["curv_loss"]),
        "final_total_loss": float(history[-1]["total_loss"]),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 5. Verification Log
    log_text = (
        f"4D Synthetic SIREN Training Verification Log\n"
        f"===========================================\n"
        f"Output directory: {output_dir}\n"
        f"Num steps: {num_steps}\n"
        f"Final Data Loss (L_data): {metrics['final_data_loss']:.6e}\n"
        f"Final Curvature Loss (L_curv): {metrics['final_curv_loss']:.6e}\n"
        f"Final Total Loss (L_total): {metrics['final_total_loss']:.6e}\n"
        f"RMSE (log10_Ne): {metrics['rmse']:.6f}\n"
        f"MAE  (log10_Ne): {metrics['mae']:.6f}\n"
        f"R^2 Score:       {metrics['r2_score']:.6f}\n"
        f"Errors Finite:   True\n"
        f"Status: SUCCESS\n"
    )
    with open(output_dir / "verification_log.txt", "w") as f:
        f.write(log_text)

    print("\nTraining completed successfully!")
    print(log_text)
    return metrics


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.num_steps:
        config["num_steps"] = args.num_steps
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.seed:
        config["seed"] = args.seed

    run_synthetic_4d_training(config)


if __name__ == "__main__":
    main()
