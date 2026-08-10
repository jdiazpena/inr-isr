"""Analytic, integrated, and observation-geometry synthetic 4D contracts."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inr_isr_4d.synthetic import (
    MovingFeature4D,
    SyntheticFieldConfig,
    SyntheticObservationConfig,
    evaluate_observation_target,
    evaluate_synthetic_truth,
    generate_synthetic_case,
    independent_truth_points,
)


@pytest.mark.parametrize("variant", ["gaussian_3d", "chapman_f2_reference"])
def test_named_variants_are_finite_and_have_analytic_first_derivatives(variant: str) -> None:
    config = SyntheticFieldConfig(variant=variant)
    coordinates = np.array(
        [[-20.0, 10.0, 250.0, 40.0], [40.0, -30.0, 320.0, 100.0]], dtype=float
    )
    values = evaluate_synthetic_truth(coordinates, config)
    for key in ("Ne", "log10_Ne", "dlog10Ne_dx", "dlog10Ne_dy", "dlog10Ne_dz", "dlog10Ne_dt"):
        assert np.all(np.isfinite(values[key]))
    epsilon = 1.0e-4
    for axis, name in enumerate(("x", "y", "z", "t")):
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[:, axis] += epsilon
        minus[:, axis] -= epsilon
        numerical = (
            evaluate_synthetic_truth(plus, config)["log10_Ne"]
            - evaluate_synthetic_truth(minus, config)["log10_Ne"]
        ) / (2 * epsilon)
        np.testing.assert_allclose(numerical, values[f"dlog10Ne_d{name}"], rtol=2e-5, atol=2e-8)


def test_instantaneous_observation_is_exact_midpoint_truth() -> None:
    coordinates = np.array([[0.0, 0.0, 300.0, 100.0], [10.0, 20.0, 250.0, 200.0]])
    field = SyntheticFieldConfig()
    truth = evaluate_synthetic_truth(coordinates, field)
    observation = evaluate_observation_target(
        coordinates,
        field,
        mode="instantaneous",
        integration_duration_sec=0.0,
        integration_samples=1,
    )
    np.testing.assert_array_equal(observation["Ne"], truth["Ne"])
    np.testing.assert_array_equal(observation["instantaneous_Ne"], truth["Ne"])


@pytest.mark.parametrize("duration", [120.0, 300.0])
def test_integration_averaged_target_uses_declared_trapezoid(duration: float) -> None:
    field = SyntheticFieldConfig(
        feature=MovingFeature4D(x0_km=-50.0, velocity_x_km_s=0.6, velocity_y_km_s=0.0)
    )
    coordinates = np.array([[0.0, 0.0, 300.0, 100.0]])
    result = evaluate_observation_target(
        coordinates,
        field,
        mode="integration_averaged",
        integration_duration_sec=duration,
        integration_samples=5,
    )
    offsets = np.linspace(-duration / 2, duration / 2, 5)
    samples = []
    for offset in offsets:
        shifted = coordinates.copy()
        shifted[:, 3] += offset
        samples.append(evaluate_synthetic_truth(shifted, field)["Ne"][0])
    expected = np.trapz(samples, offsets) / duration
    assert result["Ne"][0] == pytest.approx(expected, rel=1e-14)
    assert result["log10_Ne"][0] == pytest.approx(np.log10(expected), rel=1e-14)
    assert result["Ne"][0] != pytest.approx(result["instantaneous_Ne"][0], rel=1e-4)


@pytest.mark.parametrize(
    ("mode", "duration", "samples"),
    [("instantaneous", 0.0, 1), ("integration_averaged", 120.0, 7), ("integration_averaged", 300.0, 9)],
)
def test_generated_observation_bundle_has_exact_4d_semantics(
    mode: str, duration: float, samples: int
) -> None:
    observation = SyntheticObservationConfig(
        mode=mode,
        integration_duration_sec=duration,
        integration_samples=samples,
        n_beams=5,
        n_ranges=4,
        n_times=3,
        duration_sec=600.0,
        seed=0,
    )
    case = generate_synthetic_case(SyntheticFieldConfig(), observation)
    assert case.bundle.size == 5 * 4 * 3
    assert len(np.unique(case.bundle.beam_ids)) == 5
    assert len(np.unique(case.bundle.time_ids)) == 3
    assert np.all(case.integration_duration_sec == duration)
    np.testing.assert_allclose(
        case.integration_end_sec - case.integration_start_sec, duration
    )
    if mode == "instantaneous":
        np.testing.assert_array_equal(case.bundle.targets, case.instantaneous_log10_ne)


def test_independent_truth_sampling_is_seeded_and_separate() -> None:
    bounds = np.array([[-100, 100], [-100, 100], [150, 450], [0, 600]], dtype=float)
    first_coordinates, first_truth = independent_truth_points(
        SyntheticFieldConfig(), count=20, bounds=bounds, seed=6
    )
    second_coordinates, second_truth = independent_truth_points(
        SyntheticFieldConfig(), count=20, bounds=bounds, seed=6
    )
    np.testing.assert_array_equal(first_coordinates, second_coordinates)
    np.testing.assert_array_equal(first_truth["Ne"], second_truth["Ne"])
