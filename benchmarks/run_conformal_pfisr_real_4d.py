# -*- coding: utf-8 -*-
"""
benchmarks/run_conformal_pfisr_real_4d.py

Real 4D PFISR AMISR Held-Out Radar Beam Conformal Calibration Benchmark.
Ingests real 4D PFISR volume dataset (data/20120122.001_lp_5min.h5), splits beams into train,
calibration, and test sets via split_beams, trains 4D SIREN ONLY on train_beams, calibrates
conformal quantile q_0.95 on calib_beams, and evaluates empirical coverage and interval width on test_beams.
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
from amisr_h5_reader_4d import read_amisr_h5_4d_volume
from inr_radar.datasets.coordinate_transforms import Normalizer4D
from inr_radar.uq.conformal import split_beams, SplitConformalCalibrator
from training_common import curvature_loss_4d, compute_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real PFISR 4D Held-Out Beam Conformal Calibration")
    parser.add_argument("--config", type=str, default="configs/conformal_pfisr_random_config.json")
    parser.add_argument("--h5_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--withholding_strategy", type=str, default=None, choices=["random", "clustered"])
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def run_pfisr_real_4d_conformal_benchmark(config: dict) -> dict:
    seed = config.get("seed", 42)
    set_seed(seed)

    output_dir = Path(config.get("output_dir", "outputs/conformal_pfisr_random"))
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = Path(config.get("h5_path", "data/20120122.001_lp_5min.h5"))
    if not h5_path.is_absolute():
        inr_isr_root = Path(__file__).resolve().parent.parent
        h5_path_rel = inr_isr_root / h5_path
        if h5_path_rel.exists():
            h5_path = h5_path_rel

    print(f"Loading real 4D PFISR AMISR volume dataset: {h5_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Extract 4D PFISR Volume Dataframe
    z_min_km = config.get("z_min_km", 100.0)
    z_max_km = config.get("z_max_km", 500.0)
    time_start_utc = config.get("time_start_utc", None)
    time_end_utc = config.get("time_end_utc", None)
    window_start_index = config.get("window_start_index", 0)
    window_size_records = config.get("window_size_records", 10)

    df_full = read_amisr_h5_4d_volume(
        h5_path=h5_path,
        z_min_km=z_min_km,
        z_max_km=z_max_km,
        time_start_utc=time_start_utc,
        time_end_utc=time_end_utc,
        window_start_index=window_start_index,
        window_size_records=window_size_records,
        verbose=True,
    )

    # 2. Split Beams into Train, Calibration, and Test sets
    withholding_strategy = config.get("withholding_strategy", "random")
    calib_ratio = config.get("calib_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)
    cluster_center_xy = config.get("cluster_center_xy", [50.0, 50.0])
    cluster_radius_km = config.get("cluster_radius_km", 100.0)

    print(f"Splitting beams using strategy='{withholding_strategy}'...")
    df_train, df_calib, df_test = split_beams(
        df=df_full,
        withholding_strategy=withholding_strategy,
        calib_ratio=calib_ratio,
        test_ratio=test_ratio,
        cluster_center_xy=cluster_center_xy,
        cluster_radius_km=cluster_radius_km,
        seed=seed,
    )

    beam_col = "beamcode" if "beamcode" in df_full.columns else "beam_index"
    train_beams = np.sort(df_train[beam_col].unique())
    calib_beams = np.sort(df_calib[beam_col].unique())
    test_beams = np.sort(df_test[beam_col].unique())

    print(f"Train beams ({len(train_beams)}): {list(train_beams)}")
    print(f"Calib beams ({len(calib_beams)}): {list(calib_beams)}")
    print(f"Test beams  ({len(test_beams)}):  {list(test_beams)}")

    # 3. Normalization
    normalizer = Normalizer4D().fit(
        df_full["x_km"].to_numpy(),
        df_full["y_km"].to_numpy(),
        df_full["z_km"].to_numpy(),
        df_full["t_sec"].to_numpy(),
    )

    val_min = float(df_full["log10_Ne"].min())
    val_max = float(df_full["log10_Ne"].max())
    val_span = max(val_max - val_min, 1e-6)

    def process_df(df_sub):
        coords_phys = df_sub[["x_km", "y_km", "z_km", "t_sec"]].to_numpy(dtype=np.float32)
        coords_norm = normalizer.normalize(coords_phys)
        values_norm = 2.0 * (df_sub["log10_Ne"].to_numpy(dtype=np.float32) - val_min) / val_span - 1.0
        return (
            torch.from_numpy(coords_norm).to(dtype=torch.float32, device=device),
            torch.from_numpy(values_norm.reshape(-1, 1)).to(dtype=torch.float32, device=device),
            df_sub["log10_Ne"].to_numpy(dtype=np.float64),
        )

    train_coords, train_values, y_train_log10 = process_df(df_train)
    calib_coords, calib_values, y_calib_log10 = process_df(df_calib)
    test_coords, test_values, y_test_log10 = process_df(df_test)

    # 4. Model Training ONLY on Train Beams
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

    history = []
    print(f"Training 4D SIREN model on {len(df_train)} real PFISR train points for {num_steps} steps...")

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(train_coords)
        l_data = F.mse_loss(pred, train_values)

        l_curv, curv_details = curvature_loss_4d(
            model=model,
            coords_col=train_coords,
            loss_type=loss_type,
        )

        l_total = l_data + lambda_curv * l_curv

        if not torch.isfinite(l_total):
            raise RuntimeError(f"Step {step}: Non-finite loss: l_total={l_total.item()}")

        l_total.backward()
        optimizer.step()

        history.append({
            "step": step,
            "total_loss": l_total.item(),
            "data_loss": l_data.item(),
            "curv_loss": l_curv.item(),
        })

        if step % 100 == 0 or step == num_steps:
            print(f"Step {step:4d}/{num_steps} | Total: {l_total.item():.6e} | Data: {l_data.item():.6e}")

    # 5. Evaluate Predictions
    model.eval()
    with torch.no_grad():
        pred_calib_norm = model(calib_coords).cpu().numpy().ravel()
        pred_test_norm = model(test_coords).cpu().numpy().ravel()

    pred_calib_log10 = 0.5 * (pred_calib_norm + 1.0) * val_span + val_min
    pred_test_log10 = 0.5 * (pred_test_norm + 1.0) * val_span + val_min

    # 6. Conformal Calibration on calib_beams
    alpha = config.get("alpha", 0.05)
    calibrator = SplitConformalCalibrator(alpha=alpha)
    q_95 = calibrator.calibrate(y_true=y_calib_log10, y_pred=pred_calib_log10)

    # 7. Evaluate Empirical Coverage and Interval Width on test_beams
    conformal_eval = calibrator.evaluate_coverage(y_true=y_test_log10, y_pred=pred_test_log10)
    empirical_coverage = conformal_eval["empirical_coverage"]
    interval_width = conformal_eval["interval_width"]

    res_metrics = compute_metrics(pred_test_log10, y_test_log10)
    ss_res = np.sum((y_test_log10 - pred_test_log10) ** 2)
    ss_tot = np.sum((y_test_log10 - np.mean(y_test_log10)) ** 2)
    r2_score = float(1.0 - (ss_res / max(ss_tot, 1e-12)))

    metrics = {
        "withholding_strategy": withholding_strategy,
        "empirical_coverage": empirical_coverage,
        "q_95": q_95,
        "interval_width": interval_width,
        "target_coverage": 1.0 - alpha,
        "alpha": alpha,
        "num_train_beams": len(train_beams),
        "num_calib_beams": len(calib_beams),
        "num_test_beams": len(test_beams),
        "test_rmse": res_metrics["rmse"],
        "test_mae": res_metrics["mae"],
        "test_r2_score": r2_score,
        "final_data_loss": history[-1]["data_loss"],
        "final_total_loss": history[-1]["total_loss"],
    }

    # 8. Save Telemetry, Metrics, and Verification Log
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "conformal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "loss_telemetry.json", "w") as f:
        json.dump({"history": history, "summary": metrics}, f, indent=2)

    torch.save(model.state_dict(), output_dir / "pfisr_4d_model.pt")

    log_text = (
        f"Real PFISR 4D Conformal Calibration Verification Log\n"
        f"==================================================\n"
        f"HDF5 path:            {h5_path}\n"
        f"Withholding Strategy: {withholding_strategy}\n"
        f"Output directory:     {output_dir}\n"
        f"Train Beams:          {len(train_beams)}\n"
        f"Calib Beams:          {len(calib_beams)}\n"
        f"Test Beams:           {len(test_beams)}\n"
        f"Alpha (1 - C):        {alpha}\n"
        f"Conformal q_0.95:     {q_95:.6f}\n"
        f"Interval Width W:     {interval_width:.6f}\n"
        f"Empirical Coverage:   {empirical_coverage:.4f} (Target: {1.0 - alpha:.2f})\n"
        f"Test RMSE (log10_Ne): {res_metrics['rmse']:.6f}\n"
        f"Test MAE  (log10_Ne): {res_metrics['mae']:.6f}\n"
        f"Test R^2 Score:       {r2_score:.6f}\n"
        f"Status: SUCCESS\n"
    )

    with open(output_dir / "verification_log.txt", "w") as f:
        f.write(log_text)

    with open(output_dir / "reconstruction_log.txt", "w") as f:
        f.write(log_text)

    print("\nReal PFISR 4D Conformal Benchmark completed successfully!")
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

    if args.h5_path:
        config["h5_path"] = args.h5_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.withholding_strategy:
        config["withholding_strategy"] = args.withholding_strategy
    if args.num_steps:
        config["num_steps"] = args.num_steps
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.seed:
        config["seed"] = args.seed

    run_pfisr_real_4d_conformal_benchmark(config)


if __name__ == "__main__":
    main()
