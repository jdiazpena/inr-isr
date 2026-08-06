# -*- coding: utf-8 -*-
"""
tests/test_conformal.py

Unit tests for Conformal Prediction & Radar Beam Splitting (Phase 2):
1. Beam splitting strategies ('random' and 'clustered') with strict disjointness guarantee.
2. ConformalCalibrator4D calibration, quantile computation, prediction intervals, and empirical coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from inr_radar.uq.conformal import split_beams, ConformalCalibrator4D, SplitConformalCalibrator
from conformal import split_beams as top_split_beams
from models import MLPINR


def test_split_beams_random_disjoint():
    """Test split_beams random strategy for beam disjointness and ratio correctness."""
    n_beams = 20
    rows = []
    for b in range(n_beams):
        for r in range(5):
            rows.append({
                "beam_index": b,
                "beamcode": 1000 + b,
                "x_km": np.cos(b) * 50.0,
                "y_km": np.sin(b) * 50.0,
                "z_km": 200.0 + r * 10.0,
                "t_sec": 10.0,
                "log10_Ne": 11.5 + np.random.randn() * 0.1,
            })
    df = pd.DataFrame(rows)

    train_df, calib_df, test_df = split_beams(
        df,
        withholding_strategy="random",
        calib_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_b = set(train_df["beam_index"].unique())
    calib_b = set(calib_df["beam_index"].unique())
    test_b = set(test_df["beam_index"].unique())

    # Check strict disjointness
    assert calib_b.isdisjoint(test_b)
    assert train_b.isdisjoint(calib_b)
    assert train_b.isdisjoint(test_b)
    assert train_b.union(calib_b).union(test_b) == set(range(n_beams))

    assert len(calib_b) == int(np.round(n_beams * 0.15))
    assert len(test_b) == int(np.round(n_beams * 0.15))


def test_split_beams_clustered_disjoint():
    """Test split_beams clustered strategy for spatial beam selection and disjointness."""
    rows = []
    # Create 10 beams inside 50km radius and 10 beams outside 150km radius
    for b in range(10):
        rows.append({"beam_index": b, "x_km": float(b * 3), "y_km": float(b * 4), "log10_Ne": 11.0})
    for b in range(10, 20):
        rows.append({"beam_index": b, "x_km": float(200 + b), "y_km": float(200 + b), "log10_Ne": 11.0})
    df = pd.DataFrame(rows)

    train_df, calib_df, test_df = split_beams(
        df,
        withholding_strategy="clustered",
        calib_ratio=0.15,
        test_ratio=0.15,
        cluster_center_xy=(0.0, 0.0),
        cluster_radius_km=50.0,
        seed=123,
    )

    train_b = set(train_df["beam_index"].unique())
    calib_b = set(calib_df["beam_index"].unique())
    test_b = set(test_df["beam_index"].unique())

    assert calib_b.isdisjoint(test_b)
    assert train_b.isdisjoint(calib_b)
    assert train_b.isdisjoint(test_b)
    assert train_b.union(calib_b).union(test_b) == set(range(20))


def test_conformal_calibrator_calibration_and_intervals():
    """Test ConformalCalibrator4D calibrate, predict_interval, and evaluate_coverage."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = MLPINR(in_features=4, out_features=1, hidden_features=32, hidden_layers=2)

    # Synthetic calibration dataset: 100 samples
    calib_coords = torch.randn(100, 4)
    calib_targets = model(calib_coords).detach() + torch.randn(100, 1) * 0.05
    calib_dataset = (calib_coords, calib_targets)

    calibrator = ConformalCalibrator4D(alpha=0.05)
    q_95 = calibrator.calibrate(model, calib_dataset, device="cpu")

    assert q_95 > 0.0
    assert np.isfinite(q_95)

    # Test predict_interval
    query_coords = torch.randn(20, 4)
    y_lower, y_upper, y_pred, q_val = calibrator.predict_interval(model, query_coords, device="cpu")

    assert q_val == q_95
    assert len(y_lower) == 20
    assert len(y_upper) == 20
    assert len(y_pred) == 20
    np.testing.assert_allclose(y_upper - y_pred, q_95, atol=1e-5)
    np.testing.assert_allclose(y_pred - y_lower, q_95, atol=1e-5)

    # Test evaluate_coverage
    test_coords = torch.randn(200, 4)
    test_targets = model(test_coords).detach() + torch.randn(200, 1) * 0.05
    test_dataset = (test_coords, test_targets)

    coverage, width = calibrator.evaluate_coverage(model, test_dataset, device="cpu")

    assert 0.0 <= coverage <= 1.0
    assert width == 2.0 * q_95
    # Empirical coverage should be close to 95%
    assert coverage >= 0.85


def test_top_level_import():
    """Verify that importing from conformal.py works identically."""
    calibrator = SplitConformalCalibrator(alpha=0.05)
    assert isinstance(calibrator, ConformalCalibrator4D)
    assert top_split_beams is split_beams
