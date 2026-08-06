# -*- coding: utf-8 -*-
"""
tests/test_conformal_prediction.py

Unit tests for Held-Out Beam Conformal Prediction (Subagent 3):
a) test_random_beam_split_disjoint: Verify random withholding strategy produces disjoint train, calib, and test sets with exact ratios.
b) test_clustered_beam_split_disjoint: Verify clustered withholding strategy produces disjoint sets with localized spatial clusters.
c) test_conformal_quantile_calculation: Verify q_{1-alpha} obeys finite-sample quantile math (q_{0.95} >= 0).
d) test_conformal_coverage_on_synthetic_model: Train a 4D SIREN model on train beams, calibrate on calib beams, and verify empirical test coverage Coverage_{0.95} >= 90% on test beams.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from inr_radar.uq.conformal import (
    SplitConformalCalibrator,
    split_beams_by_strategy,
    compute_conformal_quantile,
)
from models import MLPINR
from datasets import Normalizer4D
from synthetic_plasma import SyntheticPlasma4DDataset
from training_common import set_seed


def test_random_beam_split_disjoint():
    """Verify that withholding_strategy='random' produces strictly disjoint train, calib, and test sets with correct ratios."""
    beam_ids = [f"beam_{i:02d}" for i in range(40)]
    calib_ratio = 0.15
    test_ratio = 0.15

    train_beams, calib_beams, test_beams = split_beams_by_strategy(
        beam_ids=beam_ids,
        withholding_strategy="random",
        calib_ratio=calib_ratio,
        test_ratio=test_ratio,
        seed=42,
    )

    # 1. Verify lengths and ratios
    n_total = len(beam_ids)
    expected_calib_len = int(round(n_total * calib_ratio))  # 6
    expected_test_len = int(round(n_total * test_ratio))   # 6
    expected_train_len = n_total - expected_calib_len - expected_test_len  # 28

    assert len(calib_beams) == expected_calib_len
    assert len(test_beams) == expected_test_len
    assert len(train_beams) == expected_train_len

    # 2. Verify strict disjointness
    set_train = set(train_beams)
    set_calib = set(calib_beams)
    set_test = set(test_beams)

    assert set_train.isdisjoint(set_calib), "Train and Calib beams overlap!"
    assert set_train.isdisjoint(set_test), "Train and Test beams overlap!"
    assert set_calib.isdisjoint(set_test), "Calib and Test beams overlap!"

    # 3. Verify exact union equals total beam set
    assert (set_train | set_calib | set_test) == set(beam_ids)


def test_clustered_beam_split_disjoint():
    """Verify that withholding_strategy='clustered' produces strictly disjoint sets where calibration and test beams fall within a spatial cluster."""
    set_seed(42)
    beam_ids = [f"beam_{i:02d}" for i in range(40)]

    # Create spatial coordinates for 40 beams on a 2D grid
    beam_coords = {}
    for i, b_id in enumerate(beam_ids):
        x = (i % 8) * 50.0 - 175.0  # -175 to 175 km
        y = (i // 8) * 50.0 - 100.0 # -100 to 100 km
        beam_coords[b_id] = np.array([x, y], dtype=np.float64)

    train_beams, calib_beams, test_beams = split_beams_by_strategy(
        beam_ids=beam_ids,
        withholding_strategy="clustered",
        calib_ratio=0.15,
        test_ratio=0.15,
        beam_coords=beam_coords,
        seed=123,
    )

    # 1. Verify strict disjointness
    set_train = set(train_beams)
    set_calib = set(calib_beams)
    set_test = set(test_beams)

    assert set_train.isdisjoint(set_calib)
    assert set_train.isdisjoint(set_test)
    assert set_calib.isdisjoint(set_test)
    assert (set_train | set_calib | set_test) == set(beam_ids)

    # 2. Verify spatial cluster property:
    # Measure average distance among withheld beams vs overall average beam distance
    withheld_beams = calib_beams + test_beams
    withheld_coords = np.array([beam_coords[b] for b in withheld_beams])
    all_coords = np.array([beam_coords[b] for b in beam_ids])

    # Compute mean pairwise distance within withheld cluster
    withheld_diffs = withheld_coords[:, None, :] - withheld_coords[None, :, :]
    withheld_dists = np.sqrt(np.sum(withheld_diffs ** 2, axis=-1))
    mean_withheld_dist = np.mean(withheld_dists[withheld_dists > 0])

    # Compute mean pairwise distance across all beams
    all_diffs = all_coords[:, None, :] - all_coords[None, :, :]
    all_dists = np.sqrt(np.sum(all_diffs ** 2, axis=-1))
    mean_all_dist = np.mean(all_dists[all_dists > 0])

    # Spatial cluster should have tighter mean distance than full field average
    assert mean_withheld_dist < mean_all_dist, f"Cluster mean dist ({mean_withheld_dist:.1f}) should be smaller than full field mean ({mean_all_dist:.1f})"


def test_conformal_quantile_calculation():
    """Verify that q_{1-alpha} calculation obeys finite-sample quantile math (q_{0.95} >= 0)."""
    # Case 1: Zero residuals
    zero_residuals = np.zeros(100)
    q_zero = compute_conformal_quantile(zero_residuals, alpha=0.05)
    assert q_zero == 0.0

    # Case 2: Standard non-conformity scores
    set_seed(42)
    residuals = np.abs(np.random.normal(loc=0.0, scale=0.5, size=200))
    q_95 = compute_conformal_quantile(residuals, alpha=0.05)
    assert q_95 >= 0.0

    # Verify finite sample math: fraction of residuals <= q_{0.95} must be at least (1 - alpha)
    empirical_cov_on_calib = np.mean(residuals <= q_95)
    assert empirical_cov_on_calib >= 0.95

    # Case 3: Monotonicity with alpha
    q_90 = compute_conformal_quantile(residuals, alpha=0.10)
    q_99 = compute_conformal_quantile(residuals, alpha=0.01)
    assert q_99 >= q_95 >= q_90 >= 0.0

    # Case 4: Test with PyTorch Tensors
    residuals_tensor = torch.tensor(residuals, dtype=torch.float32)
    calibrator = SplitConformalCalibrator(alpha=0.05)
    q_fit = calibrator.fit(y_calib=residuals_tensor, y_pred_calib=torch.zeros_like(residuals_tensor))
    assert q_fit >= 0.0
    assert abs(q_fit - q_95) < 1e-5


def test_conformal_coverage_on_synthetic_model():
    """Train a quick 4D SIREN model on synthetic train beams, calibrate on calib beams, and verify empirical test coverage Coverage_{0.95} >= 90% on test beams."""
    set_seed(42)

    # 1. Generate synthetic 4D dataset with 40 distinct beam assignments
    n_beams = 40
    points_per_beam = 50
    total_points = n_beams * points_per_beam

    dataset = SyntheticPlasma4DDataset(
        num_points=total_points,
        x_bounds=(-300.0, 300.0),
        y_bounds=(-300.0, 300.0),
        z_bounds=(100.0, 500.0),
        t_bounds=(0.0, 300.0),
        v_km_s=1.0,
        seed=42,
    )

    df = dataset.df.copy()
    beam_assignments = [f"beam_{i % n_beams:02d}" for i in range(len(df))]
    df["beam_id"] = beam_assignments

    # 2. Split beams into train, calib, and test sets
    unique_beams = [f"beam_{i:02d}" for i in range(n_beams)]
    train_beams, calib_beams, test_beams = split_beams_by_strategy(
        beam_ids=unique_beams,
        withholding_strategy="random",
        calib_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # Filter dataframes
    train_df = df[df["beam_id"].isin(train_beams)]
    calib_df = df[df["beam_id"].isin(calib_beams)]
    test_df = df[df["beam_id"].isin(test_beams)]

    # Normalize coordinates using Normalizer4D
    normalizer = dataset.normalizer

    train_coords = torch.tensor(normalizer.forward(train_df[["x_km", "y_km", "z_km", "t_sec"]].values), dtype=torch.float32)
    train_values = torch.tensor(train_df["log10_Ne"].values.reshape(-1, 1), dtype=torch.float32)

    calib_coords = torch.tensor(normalizer.forward(calib_df[["x_km", "y_km", "z_km", "t_sec"]].values), dtype=torch.float32)
    calib_values = torch.tensor(calib_df["log10_Ne"].values.reshape(-1, 1), dtype=torch.float32)

    test_coords = torch.tensor(normalizer.forward(test_df[["x_km", "y_km", "z_km", "t_sec"]].values), dtype=torch.float32)
    test_values = torch.tensor(test_df["log10_Ne"].values.reshape(-1, 1), dtype=torch.float32)

    # 3. Train quick 4D SIREN model on train beams
    model = MLPINR(
        in_features=4,
        out_features=1,
        hidden_features=64,
        hidden_layers=3,
        activation="sine",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    for step in range(50):
        optimizer.zero_grad()
        pred = model(train_coords)
        loss = torch.nn.functional.mse_loss(pred, train_values)
        loss.backward()
        optimizer.step()

    # 4. Calibrate SplitConformalCalibrator on held-out calib beams
    model.eval()
    with torch.no_grad():
        calib_pred = model(calib_coords)
        test_pred = model(test_coords)

    calibrator = SplitConformalCalibrator(alpha=0.05)
    q_hat = calibrator.fit(y_calib=calib_values, y_pred_calib=calib_pred)
    assert q_hat >= 0.0

    # 5. Evaluate coverage on held-out test beams
    coverage = calibrator.evaluate_coverage(y_true=test_values, y_pred=test_pred)

    # Coverage threshold requirement: >= 90% (0.90)
    assert coverage >= 0.90, f"Empirical coverage on test beams is {coverage:.3f}, expected >= 0.90"
