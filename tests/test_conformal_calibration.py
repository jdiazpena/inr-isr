# -*- coding: utf-8 -*-
"""
tests/test_conformal_calibration.py

Unit tests for Conformal Prediction UQ in inr-isr:
a) split_beams with 'random' and 'clustered' withholding strategies.
b) SplitConformalCalibrator quantile calibration, prediction intervals, and empirical coverage.
c) End-to-end benchmark execution sanity check for synthetic and real drivers.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from inr_radar.datasets.synthetic_generator_4d import make_observation_geometry_4d, generate_synthetic_beam_dataset_4d
from inr_radar.uq.conformal import split_beams, SplitConformalCalibrator
from benchmarks.run_conformal_synthetic_4d import run_synthetic_4d_conformal_benchmark
from benchmarks.run_conformal_pfisr_real_4d import run_pfisr_real_4d_conformal_benchmark


def test_split_beams_random_strategy():
    """Test split_beams using random withholding strategy."""
    df_geom = make_observation_geometry_4d(n_beams=40, seed=42)
    df_geom["log10_Ne"] = 11.0 + 0.1 * np.random.randn(len(df_geom))

    train_df, calib_df, test_df = split_beams(
        df=df_geom,
        withholding_strategy="random",
        calib_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_beams = set(train_df["beamcode"].unique())
    calib_beams = set(calib_df["beamcode"].unique())
    test_beams = set(test_df["beamcode"].unique())

    # Ensure sets are disjoint
    assert train_beams.isdisjoint(calib_beams)
    assert train_beams.isdisjoint(test_beams)
    assert calib_beams.isdisjoint(test_beams)

    # Ensure all beams are accounted for
    all_split_beams = train_beams.union(calib_beams).union(test_beams)
    assert all_split_beams == set(df_geom["beamcode"].unique())

    # Check non-empty subsets
    assert len(train_df) > 0
    assert len(calib_df) > 0
    assert len(test_df) > 0


def test_split_beams_clustered_strategy():
    """Test split_beams using clustered spatial withholding strategy."""
    df_geom = make_observation_geometry_4d(n_beams=42, seed=42)
    df_geom["log10_Ne"] = 11.0 + 0.1 * np.random.randn(len(df_geom))

    train_df, calib_df, test_df = split_beams(
        df=df_geom,
        withholding_strategy="clustered",
        cluster_center_xy=[0.0, 0.0],
        cluster_radius_km=150.0,
        seed=42,
    )

    train_beams = set(train_df["beamcode"].unique())
    calib_beams = set(calib_df["beamcode"].unique())
    test_beams = set(test_df["beamcode"].unique())

    # Ensure sets are disjoint
    assert train_beams.isdisjoint(calib_beams)
    assert train_beams.isdisjoint(test_beams)
    assert calib_beams.isdisjoint(test_beams)

    # All beams present
    assert train_beams.union(calib_beams).union(test_beams) == set(df_geom["beamcode"].unique())

    assert len(train_df) > 0
    assert len(calib_df) > 0
    assert len(test_df) > 0


def test_split_conformal_calibrator():
    """Test SplitConformalCalibrator quantile calculation, intervals, and coverage evaluation."""
    rng = np.random.default_rng(42)

    # Synthetic calibration residuals with standard deviation 0.5
    y_true_calib = 11.0 + rng.normal(0, 0.5, size=200)
    y_pred_calib = y_true_calib + rng.normal(0, 0.1, size=200)

    calibrator = SplitConformalCalibrator(alpha=0.05)
    q_95 = calibrator.calibrate(y_true=y_true_calib, y_pred=y_pred_calib)

    assert q_95 > 0.0
    assert np.isfinite(q_95)

    # Test set evaluation
    y_true_test = 11.0 + rng.normal(0, 0.5, size=100)
    y_pred_test = y_true_test + rng.normal(0, 0.1, size=100)

    lower, upper = calibrator.predict_interval(y_pred_test)
    assert len(lower) == 100
    assert len(upper) == 100
    assert np.all(upper > lower)
    np.testing.assert_allclose(upper - lower, 2.0 * q_95)

    conformal_eval = calibrator.evaluate_coverage(y_true=y_true_test, y_pred=y_pred_test)
    assert conformal_eval["empirical_coverage"] >= 0.90
    assert conformal_eval["q_95"] == q_95
    assert conformal_eval["interval_width"] == 2.0 * q_95


def test_run_synthetic_4d_conformal_benchmark_sanity(tmp_path):
    """Sanity check for run_synthetic_4d_conformal_benchmark execution."""
    config = {
        "dataset_type": "synthetic_4d",
        "in_features": 4,
        "out_features": 1,
        "withholding_strategy": "random",
        "calib_ratio": 0.15,
        "test_ratio": 0.15,
        "alpha": 0.05,
        "n_beams": 20,
        "n_ranges": 10,
        "n_times": 5,
        "z_min_km": 100.0,
        "z_max_km": 500.0,
        "hidden_features": 32,
        "hidden_layers": 2,
        "learning_rate": 1e-3,
        "num_steps": 10,
        "seed": 42,
        "output_dir": str(tmp_path / "synthetic_conformal_test"),
    }

    metrics = run_synthetic_4d_conformal_benchmark(config)
    assert "empirical_coverage" in metrics
    assert "q_95" in metrics
    assert "interval_width" in metrics
    assert metrics["empirical_coverage"] >= 0.0
    assert metrics["q_95"] > 0.0
    assert (tmp_path / "synthetic_conformal_test" / "verification_log.txt").exists()


def test_run_pfisr_real_4d_conformal_benchmark_sanity(tmp_path):
    """Sanity check for run_pfisr_real_4d_conformal_benchmark execution with real HDF5 dataset."""
    h5_path = Path("data/20120122.001_lp_5min.h5")
    if not h5_path.is_absolute():
        inr_isr_root = Path(__file__).resolve().parent.parent
        h5_path = inr_isr_root / h5_path

    if not h5_path.exists():
        pytest.skip(f"PFISR HDF5 dataset not found at {h5_path}")

    config = {
        "h5_path": str(h5_path),
        "dataset_type": "pfisr_real_4d",
        "in_features": 4,
        "out_features": 1,
        "withholding_strategy": "clustered",
        "cluster_center_xy": [50.0, 50.0],
        "cluster_radius_km": 100.0,
        "calib_ratio": 0.15,
        "test_ratio": 0.15,
        "alpha": 0.05,
        "z_min_km": 100.0,
        "z_max_km": 500.0,
        "window_start_index": 0,
        "window_size_records": 3,
        "hidden_features": 32,
        "hidden_layers": 2,
        "learning_rate": 1e-3,
        "num_steps": 10,
        "seed": 42,
        "output_dir": str(tmp_path / "pfisr_conformal_test"),
    }

    metrics = run_pfisr_real_4d_conformal_benchmark(config)
    assert "empirical_coverage" in metrics
    assert "q_95" in metrics
    assert "interval_width" in metrics
    assert metrics["empirical_coverage"] >= 0.0
    assert metrics["q_95"] > 0.0
    assert (tmp_path / "pfisr_conformal_test" / "verification_log.txt").exists()
