"""Strict configuration and operational collocation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inr_isr_4d.collocation import make_collocation_pool
from inr_isr_4d.config import FourDConfig, apply_explicit_overrides, config_from_mapping


def test_unknown_configuration_fields_are_rejected_at_every_level() -> None:
    with pytest.raises(ValueError, match="root"):
        config_from_mapping({"mystery": {}})
    with pytest.raises(ValueError, match="model"):
        config_from_mapping({"model": {"mystery": 2}})


def test_explicit_zero_seed_override_is_not_lost() -> None:
    base = config_from_mapping({"optimization": {"seed": 9}})
    updated = apply_explicit_overrides(base, {"optimization.seed": 0})
    assert updated.optimization.seed == 0


@pytest.mark.parametrize(
    "values",
    [
        {"optimization": {"learning_rate": float("nan")}},
        {"collocation": {"pool_size": 4, "batch_size": 5}},
        {"collocation": {"batch_size": 4, "derivative_microbatch_size": 5}},
        {"derivative_prior": {"mode": "none", "weight": 1.0}},
        {"runtime": {"device": "gpu"}},
        {"runtime": {"precision": "float64", "amp": True}},
    ],
)
def test_invalid_configuration_combinations_are_rejected(values: dict) -> None:
    with pytest.raises(ValueError):
        config_from_mapping(values)


def test_default_configuration_round_trip_is_stable() -> None:
    config = FourDConfig().validate()
    assert config_from_mapping(config.to_dict()) == config


@pytest.mark.parametrize("mode", ["data_coordinates", "sobol", "support_aware"])
def test_collocation_modes_have_exact_reproducible_sizes(mode: str) -> None:
    data = torch.tensor(
        [[-1.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    )
    first = make_collocation_pool(
        mode=mode, pool_size=17, seed=0, data_coordinates=data, support_oversample_factor=3
    )
    second = make_collocation_pool(
        mode=mode, pool_size=17, seed=0, data_coordinates=data, support_oversample_factor=3
    )
    assert len(first) == 17
    torch.testing.assert_close(first.coordinates, second.coordinates, rtol=0, atol=0)
    assert torch.all(first.coordinates >= -1.0)
    assert torch.all(first.coordinates <= 1.0)


def test_pool_size_and_batch_size_change_actual_derivative_sets() -> None:
    small = make_collocation_pool(mode="sobol", pool_size=8, seed=4)
    large = make_collocation_pool(mode="sobol", pool_size=16, seed=4)
    assert len(small) == 8 and len(large) == 16
    assert small.sample(3, step=1).shape == (3, 4)
    assert large.sample(7, step=1).shape == (7, 4)
    torch.testing.assert_close(small.coordinates, large.coordinates[:8], rtol=0, atol=0)


def test_declared_collocation_domain_bounds_are_consumed() -> None:
    config = config_from_mapping(
        {
            "collocation": {
                "domain_lower": [-0.5, -0.25, 0.0, 0.25],
                "domain_upper": [0.5, 0.25, 0.75, 1.0],
            }
        }
    )
    pool = make_collocation_pool(
        mode="sobol",
        pool_size=64,
        seed=2,
        domain_lower=config.collocation.domain_lower,
        domain_upper=config.collocation.domain_upper,
    )
    lower = torch.tensor(config.collocation.domain_lower)
    upper = torch.tensor(config.collocation.domain_upper)
    assert torch.all(pool.coordinates >= lower)
    assert torch.all(pool.coordinates <= upper)
    with pytest.raises(ValueError, match="normalized collocation bound"):
        config_from_mapping(
            {"collocation": {"domain_lower": [-1.1, -1.0, -1.0, -1.0]}}
        )


def test_collocation_batch_changes_deterministically_with_step() -> None:
    pool = make_collocation_pool(mode="sobol", pool_size=32, seed=2)
    same_a = pool.sample(8, step=5)
    same_b = pool.sample(8, step=5)
    different = pool.sample(8, step=6)
    torch.testing.assert_close(same_a, same_b, rtol=0, atol=0)
    assert not torch.equal(same_a, different)
