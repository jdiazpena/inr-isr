# -*- coding: utf-8 -*-
"""
benchmarks/train_pfisr_real_4d.py

Train a 4D SIREN model on real multi-altitude PFISR AMISR database volume observations.
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
from amisr_h5_reader_4d import PFISRVolume4DDataset
from training_common import curvature_loss_4d, compute_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 4D SIREN on real multi-altitude PFISR AMISR data")
    parser.add_argument("--config", type=str, default="configs/pfisr_real_4d_config.json")
    parser.add_argument("--h5_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def run_pfisr_real_4d_training(config: dict) -> dict:
    seed = config.get("seed", 42)
    set_seed(seed)

    output_dir = Path(config.get("output_dir", "outputs/pfisr_real_4d_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = Path(config.get("h5_path", "data/20120122.001_lp_5min.h5"))
    if not h5_path.is_absolute():
        # Check relative to inr-isr root
        inr_isr_root = Path(__file__).resolve().parent.parent
        h5_path_rel = inr_isr_root / h5_path
        if h5_path_rel.exists():
            h5_path = h5_path_rel

    print(f"Loading real PFISR AMISR database HDF5 file: {h5_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    dataset = PFISRVolume4DDataset(
        h5_path=h5_path,
        z_min_km=config.get("z_min_km", 100.0),
        z_max_km=config.get("z_max_km", 500.0),
        time_start_utc=config.get("time_start_utc", None),
        time_end_utc=config.get("time_end_utc", None),
        window_start_index=config.get("window_start_index", 0),
        window_size_records=config.get("window_size_records", 10),
        verbose=True,
    )

    sample = dataset[0]
    coords = sample["coords"].to(device)
    values = sample["values"].to(device)

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

    history = []
    print(f"Starting 4D real PFISR volume training for {num_steps} steps...")

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(coords)
        l_data = F.mse_loss(pred, values)

        # Autograd 4D spatial-temporal curvature loss
        l_curv, curv_details = curvature_loss_4d(model, collocation_pool)

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
            "curv_zz": curv_details["curv_zz"].item(),
            "curv_tt": curv_details["curv_tt"].item(),
        }
        history.append(step_dict)

        if step % 100 == 0 or step == num_steps:
            print(
                f"Step {step:4d}/{num_steps} | Total Loss: {l_total.item():.6e} | "
                f"Data Loss: {l_data.item():.6e} | Curv Loss: {l_curv.item():.6e}"
            )

    # 3. Save Telemetry & Checkpoints
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    torch.save(model.state_dict(), output_dir / "pfisr_4d_model.pt")
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

    telemetry = {
        "num_observations": len(dataset),
        "altitude_range_km": [config.get("z_min_km", 100.0), config.get("z_max_km", 500.0)],
        "final_data_loss": float(history[-1]["data_loss"]),
        "final_curv_loss": float(history[-1]["curv_loss"]),
        "final_total_loss": float(history[-1]["total_loss"]),
        "mse": res_metrics["mse"],
        "rmse": res_metrics["rmse"],
        "mae": res_metrics["mae"],
        "r2_score": r2_score,
        "errors_finite": bool(np.all(np.isfinite(pred_log10_ne))),
    }

    with open(output_dir / "loss_telemetry.json", "w") as f:
        json.dump(telemetry, f, indent=2)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(telemetry, f, indent=2)

    # 5. Reconstruction Logs
    log_text = (
        f"Real 4D PFISR AMISR SIREN Training Reconstruction Log\n"
        f"====================================================\n"
        f"HDF5 path:        {h5_path}\n"
        f"Output directory: {output_dir}\n"
        f"Observations:     {len(dataset)}\n"
        f"Num steps:        {num_steps}\n"
        f"Final Data Loss (L_data): {telemetry['final_data_loss']:.6e}\n"
        f"Final Curv Loss (L_curv): {telemetry['final_curv_loss']:.6e}\n"
        f"Final Total Loss (L_total): {telemetry['final_total_loss']:.6e}\n"
        f"RMSE (log10_Ne):  {telemetry['rmse']:.6f}\n"
        f"MAE  (log10_Ne):  {telemetry['mae']:.6f}\n"
        f"R^2 Score:        {telemetry['r2_score']:.6f}\n"
        f"Errors Finite:    {telemetry['errors_finite']}\n"
        f"Status: SUCCESS\n"
    )
    with open(output_dir / "reconstruction_log.txt", "w") as f:
        f.write(log_text)

    with open(output_dir / "verification_log.txt", "w") as f:
        f.write(log_text)

    print("\nReal PFISR 4D training completed successfully!")
    print(log_text)
    return telemetry


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    if args.h5_path:
        config["h5_path"] = args.h5_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.num_steps:
        config["num_steps"] = args.num_steps
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.seed:
        config["seed"] = args.seed

    run_pfisr_real_4d_training(config)


if __name__ == "__main__":
    main()
