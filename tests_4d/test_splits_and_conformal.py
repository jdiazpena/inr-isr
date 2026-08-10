"""Leakage, exact group split, preprocessing, and conformal tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inr_isr_4d.conformal import (
    absolute_residual_scores,
    calibrate_split_conformal,
    conformal_intervals,
    empirical_interval_metrics,
    finite_sample_quantile,
    group_bootstrap_coverage,
    stratified_interval_metrics,
)
from inr_isr_4d.data import FieldBundle4D
from inr_isr_4d.config import (
    CollocationConfig,
    DerivativePriorConfig,
    FourDConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
)
from inr_isr_4d.evaluation import (
    calibrate_and_evaluate,
    calibrate_checkpoint,
    evaluate_checkpoint,
)
from inr_isr_4d.splits import make_group_split, observation_group_ids, prepare_training_problem
from inr_isr_4d.training import train_4d


def make_grouped_bundle() -> FieldBundle4D:
    rows = []
    targets = []
    beams = []
    times = []
    for time in range(5):
        for beam in range(6):
            rows.append([10.0 * beam, 3.0 * beam, 200.0 + 5.0 * time, 120.0 * time])
            targets.append([10.0 + 0.1 * beam + 0.01 * time])
            beams.append(100 + beam)
            times.append(time)
    return FieldBundle4D(
        coordinates=np.asarray(rows),
        targets=np.asarray(targets),
        beam_ids=np.asarray(beams),
        time_ids=np.asarray(times),
        group_ids=np.array([f"{beam}:{time}" for beam, time in zip(beams, times)]),
    )


@pytest.mark.parametrize("unit", ["beam", "time_block", "beam_time"])
@pytest.mark.parametrize("strategy", ["random", "clustered"])
def test_group_splits_are_exact_deterministic_and_disjoint(unit: str, strategy: str) -> None:
    bundle = make_grouped_bundle()
    first = make_group_split(
        bundle,
        unit=unit,
        strategy=strategy,
        validation_group_count=1,
        calibration_group_count=1,
        test_group_count=1,
        seed=0,
    )
    second = make_group_split(
        bundle,
        unit=unit,
        strategy=strategy,
        validation_group_count=1,
        calibration_group_count=1,
        test_group_count=1,
        seed=0,
    )
    assert first.state_dict() == second.state_dict()
    assert len(first.validation_groups) == len(first.calibration_groups) == len(first.test_groups) == 1
    all_groups = [
        set(first.train_groups),
        set(first.validation_groups),
        set(first.calibration_groups),
        set(first.test_groups),
    ]
    for index, left in enumerate(all_groups):
        for right in all_groups[index + 1 :]:
            assert left.isdisjoint(right)
    all_indices = np.concatenate(
        [first.train_indices, first.validation_indices, first.calibration_indices, first.test_indices]
    )
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(bundle.size))
    for role in ("train", "validation", "calibration", "test"):
        assert len(first.group_geometry[role]) == len(getattr(first, f"{role}_groups"))
        assert all("nearest_training_group_normalized_centroid_distance" in row for row in first.group_geometry[role])


def test_train_only_preprocessing_does_not_clip_held_out_extrema() -> None:
    bundle = make_grouped_bundle()
    split = make_group_split(
        bundle,
        unit="beam",
        strategy="clustered",
        validation_group_count=0,
        calibration_group_count=1,
        test_group_count=1,
        seed=2,
    )
    problem = prepare_training_problem(bundle, split)
    np.testing.assert_allclose(
        problem.target_scaler.minimum,
        bundle.targets[split.train_indices].min(axis=0),
    )
    np.testing.assert_allclose(
        problem.target_scaler.maximum,
        bundle.targets[split.train_indices].max(axis=0),
    )
    transformed_held = problem.target_scaler.transform(
        bundle.targets[np.concatenate([split.calibration_indices, split.test_indices])]
    )
    assert np.any((transformed_held < -1.0) | (transformed_held > 1.0))


@pytest.mark.parametrize(
    ("scores", "alpha", "expected_rank", "expected_quantile"),
    [
        ([1.0, 4.0, 2.0, 3.0], 0.4, 3, 3.0),
        ([1.0, 4.0, 2.0, 3.0], 0.1, 4, 4.0),
        ([5.0], 0.5, 1, 5.0),
    ],
)
def test_finite_sample_quantile_uses_exact_upper_rank(
    scores: list[float], alpha: float, expected_rank: int, expected_quantile: float
) -> None:
    quantile, rank = finite_sample_quantile(np.asarray(scores), alpha)
    assert rank == expected_rank and quantile == expected_quantile


@pytest.mark.parametrize(
    "call",
    [
        lambda: finite_sample_quantile(np.array([]), 0.1),
        lambda: finite_sample_quantile(np.array([1.0, np.nan]), 0.1),
        lambda: finite_sample_quantile(np.array([1.0]), 0.0),
        lambda: absolute_residual_scores(np.array([1.0]), np.array([[1.0]])),
        lambda: absolute_residual_scores(np.array([np.inf]), np.array([1.0])),
    ],
)
def test_conformal_invalid_inputs_never_fall_back_to_zero_width(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_calibration_intervals_strata_and_group_bootstrap() -> None:
    calibration = calibrate_split_conformal(
        predictions=np.array([0.0, 1.0, 2.0, 3.0]),
        targets=np.array([0.2, 1.1, 2.4, 2.7]),
        alpha=0.25,
        calibration_groups=("b1", "b2"),
        calibration_unit="beam",
        model_identity="sha256:abc",
        prediction_transform="inverse training-only affine log10Ne",
    )
    assert calibration.rank == 4 and calibration.quantile == pytest.approx(0.4)
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.2, 2.5, 2.8, 4.6])
    lower, upper = conformal_intervals(predictions, calibration)
    metrics = empirical_interval_metrics(targets, lower, upper)
    assert metrics["empirical_coverage"] == 0.5
    strata = stratified_interval_metrics(
        targets,
        lower,
        upper,
        {"beam": np.array([1, 1, 2, 2]), "altitude_bin": np.array(["low", "high", "low", "high"])},
    )
    assert strata["beam"]["1"]["count"] == 2
    bootstrap = group_bootstrap_coverage(
        targets, lower, upper, np.array([1, 1, 2, 2]), repetitions=20, seed=0
    )
    assert bootstrap["repetitions"] == 20
    assert 0 <= bootstrap["p025"] <= bootstrap["p975"] <= 1


def test_checkpoint_consuming_calibration_and_evaluation_stage(tmp_path: Path) -> None:
    bundle = make_grouped_bundle()
    split = make_group_split(
        bundle,
        unit="beam",
        strategy="random",
        validation_group_count=0,
        calibration_group_count=2,
        test_group_count=2,
        seed=0,
    )
    problem = prepare_training_problem(bundle, split)
    config = FourDConfig(
        model=ModelConfig(hidden_features=8, hidden_layers=1),
        optimization=OptimizationConfig(
            learning_rate=1.0e-3, num_steps=2, data_batch_size=10, seed=0
        ),
        collocation=CollocationConfig(
            mode="sobol", pool_size=8, batch_size=4, derivative_microbatch_size=2
        ),
        derivative_prior=DerivativePriorConfig(),
        runtime=RuntimeConfig(
            device="cpu",
            inference_chunk_size=3,
            diagnostic_probe_size=4,
            history_every=1,
            checkpoint_every=1,
        ),
    ).validate()
    run = train_4d(problem, config, tmp_path / "train")
    summary = calibrate_and_evaluate(
        bundle=bundle,
        split=split,
        coordinate_scaler=problem.coordinate_scaler,
        target_scaler=problem.target_scaler,
        checkpoint_path=run.output_directory / "checkpoint.pt",
        output_directory=tmp_path / "evaluation",
        alpha=0.1,
        bootstrap_repetitions=10,
    )
    assert summary["interpretation"]["coverage"].startswith("empirical")
    assert summary["marginal_interval_metrics"]["count"] == len(split.test_indices)
    assert (tmp_path / "evaluation" / "COMPLETED.json").is_file()
    assert (tmp_path / "evaluation" / "metrics.csv").is_file()
    assert (tmp_path / "evaluation" / "stratified_intervals.csv").is_file()
    predictions = np.load(tmp_path / "evaluation" / "predictions.npz")
    assert set(predictions.files) == {
        "coordinates",
        "targets",
        "predictions",
        "residuals",
        "interval_lower",
        "interval_upper",
        "beam_ids",
        "time_ids",
        "group_ids",
        "support_distance_km",
    }

    separate_calibration = tmp_path / "separate" / "calibration.json"
    state = calibrate_checkpoint(
        bundle=bundle,
        split=split,
        coordinate_scaler=problem.coordinate_scaler,
        target_scaler=problem.target_scaler,
        checkpoint_path=run.output_directory / "checkpoint.pt",
        calibration_path=separate_calibration,
        alpha=0.1,
    )
    assert state["rank_one_indexed"] >= 1
    separate_summary = evaluate_checkpoint(
        bundle=bundle,
        split=split,
        coordinate_scaler=problem.coordinate_scaler,
        target_scaler=problem.target_scaler,
        checkpoint_path=run.output_directory / "checkpoint.pt",
        calibration_path=separate_calibration,
        output_directory=tmp_path / "separate" / "evaluation",
        bootstrap_repetitions=10,
    )
    assert separate_summary["marginal_interval_metrics"] == summary["marginal_interval_metrics"]
