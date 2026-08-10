"""Characterization tests for the first additive 4D milestone."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inr_isr_4d.data import AffineScaler, FieldBundle4D, SampleDataset4D
from inr_isr_4d.model import SIREN4D
from inr_isr_4d.regularization import (
    PhysicalCoordinateScale,
    derivative_prior_4d,
    second_derivatives_4d,
)
from models import MLPINR


class AnalyticQuadratic4D(torch.nn.Module):
    """Field with nonzero diagonal and all mixed spatial derivatives."""

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x, y, z, t = (coordinates[:, i : i + 1] for i in range(4))
        return x**2 + 2 * y**2 + 3 * z**2 + 4 * t**2 + 5 * x * y + 6 * x * z + 7 * y * z


def make_bundle() -> FieldBundle4D:
    return FieldBundle4D(
        coordinates=np.array(
            [[0.0, 10.0, 100.0, 0.0], [1.0, 11.0, 110.0, 60.0], [2.0, 12.0, 120.0, 120.0]]
        ),
        targets=np.array([[10.0], [11.0], [20.0]]),
        beam_ids=np.array([101, 102, 103]),
        time_ids=np.array([0, 1, 2]),
        group_ids=np.array(["101:0", "102:1", "103:2"]),
        metadata={"source": "unit-test"},
    )


def test_siren4d_is_exact_four_input_specialization() -> None:
    torch.manual_seed(17)
    direct = MLPINR(in_features=4, hidden_features=12, hidden_layers=1)
    torch.manual_seed(17)
    specialized = SIREN4D(hidden_features=12, hidden_layers=1)
    direct.load_state_dict(specialized.state_dict())
    coordinates = torch.randn(7, 4, requires_grad=True)
    expected = direct(coordinates)
    actual = specialized(coordinates)
    assert actual.shape == (7, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    first = torch.autograd.grad(actual.sum(), coordinates, create_graph=True)[0]
    second_x = torch.autograd.grad(first[:, 0].sum(), coordinates)[0][:, 0]
    assert first.shape == (7, 4)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second_x).all()


def test_bundle_is_explicit_and_dataset_has_conventional_indexing() -> None:
    bundle = make_bundle()
    coordinate_scaler = AffineScaler.fit(bundle.coordinates[:2])
    target_scaler = AffineScaler.fit(bundle.targets[:2])
    dataset = SampleDataset4D(bundle, coordinate_scaler, target_scaler)
    assert bundle.size == len(dataset) == 3
    assert dataset[0]["coords"].shape == (4,)
    assert dataset[0]["values"].shape == (1,)
    assert dataset[2]["index"].item() == 2
    assert dataset[2]["coords"][0].item() > 1.0
    assert dataset[2]["values"].item() > 1.0
    with pytest.raises(IndexError):
        _ = dataset[3]
    with pytest.raises(ValueError):
        bundle.coordinates[0, 0] = 4.0


def test_scalers_are_fit_only_on_explicit_training_values() -> None:
    training = np.array([[10.0], [12.0]])
    scaler = AffineScaler.fit(training)
    transformed = scaler.transform(np.array([[9.0], [10.0], [12.0], [14.0]]))
    np.testing.assert_allclose(transformed[:, 0], [-2.0, -1.0, 1.0, 3.0])
    np.testing.assert_allclose(scaler.inverse_transform(transformed), [[9.0], [10.0], [12.0], [14.0]])


def test_all_named_derivatives_match_analytic_field() -> None:
    coordinates = torch.randn(5, 4, dtype=torch.float64)
    derivatives = second_derivatives_4d(AnalyticQuadratic4D(), coordinates)
    expected = {"xx": 2.0, "yy": 4.0, "zz": 6.0, "tt": 8.0, "xy": 5.0, "xz": 6.0, "yz": 7.0}
    for name, value in expected.items():
        torch.testing.assert_close(derivatives[name], torch.full((5, 1), value, dtype=torch.float64))


def test_named_derivative_prior_formulations_are_exact() -> None:
    coordinates = torch.randn(5, 4, dtype=torch.float64)
    legacy, _ = derivative_prior_4d(
        AnalyticQuadratic4D(), coordinates, mode="legacy_diagonal_4d"
    )
    anisotropic, _ = derivative_prior_4d(
        AnalyticQuadratic4D(),
        coordinates,
        mode="anisotropic_huber_4d",
        weight_horizontal=2.0,
        weight_vertical=3.0,
        weight_temporal=4.0,
        huber_delta_vertical=0.1,
    )
    spatial, _ = derivative_prior_4d(
        AnalyticQuadratic4D(), coordinates, mode="spatial_hessian_3d"
    )
    assert legacy.item() == pytest.approx(120.0)
    assert anisotropic.item() == pytest.approx(2.0 * 70.0 + 3.0 * 0.595 + 4.0 * 64.0)
    assert spatial.item() == pytest.approx(276.0 + 64.0)


def test_physical_coordinate_chain_rule_scaling() -> None:
    coordinates = torch.randn(4, 4, dtype=torch.float64)
    derivatives = second_derivatives_4d(
        AnalyticQuadratic4D(),
        coordinates,
        physical_scale=PhysicalCoordinateScale(x_km=4.0, y_km=8.0, z_km=10.0, t_sec=20.0),
    )
    assert derivatives["xx"][0].item() == pytest.approx(0.5)
    assert derivatives["xy"][0].item() == pytest.approx(0.625)
    assert derivatives["tt"][0].item() == pytest.approx(0.08)


@pytest.mark.parametrize("shape", [(3,), (3, 3), (0, 4)])
def test_derivative_input_contract_rejects_invalid_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        second_derivatives_4d(AnalyticQuadratic4D(), torch.empty(shape))
