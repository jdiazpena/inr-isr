# -*- coding: utf-8 -*-
"""
tests/test_4d_expansion.py

Unit tests for Phase 2 4D Expansion in inr-isr:
a) Normalizer4D forward and inverse coordinate transforms.
b) 4D Synthetic generator sampling.
c) 4D SIREN model forward pass and PyTorch autograd gradient calculation.
d) Synthetic 4D training step sanity check ensuring loss decreases and errors remain finite.
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

from models import MLPINR
from datasets import Normalizer4D
from synthetic_plasma import (
    MovingGaussianPatch4D,
    evaluate_synthetic_plasma_4d,
    SyntheticPlasma4DDataset,
)
from training_common import curvature_loss_4d, set_seed


def test_normalizer_4d_forward_inverse():
    """Test Normalizer4D forward and inverse coordinate transforms on NumPy and PyTorch tensors."""
    bounds = {
        "x_km": (-300.0, 300.0),
        "y_km": (-200.0, 200.0),
        "z_km": (100.0, 500.0),
        "t_sec": (0.0, 300.0),
    }
    normalizer = Normalizer4D(bounds=bounds)

    # 1. NumPy arrays
    raw_coords_np = np.array([
        [-300.0, -200.0, 100.0, 0.0],
        [300.0, 200.0, 500.0, 300.0],
        [0.0, 0.0, 300.0, 150.0],
    ], dtype=np.float64)

    norm_coords_np = normalizer.forward(raw_coords_np)
    # Bounds should map to -1 and +1
    np.testing.assert_allclose(norm_coords_np[0], [-1.0, -1.0, -1.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(norm_coords_np[1], [1.0, 1.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(norm_coords_np[2], [0.0, 0.0, 0.0, 0.0], atol=1e-5)

    rec_coords_np = normalizer.inverse(norm_coords_np)
    np.testing.assert_allclose(rec_coords_np, raw_coords_np, atol=1e-5)

    # 2. PyTorch Tensors
    raw_coords_torch = torch.tensor(raw_coords_np, dtype=torch.float32)
    norm_coords_torch = normalizer.forward(raw_coords_torch)
    assert isinstance(norm_coords_torch, torch.Tensor)
    rec_coords_torch = normalizer.inverse(norm_coords_torch)
    torch.testing.assert_close(rec_coords_torch, raw_coords_torch, atol=1e-4, rtol=1e-4)


def test_4d_synthetic_generator_sampling():
    """Test 4D synthetic plasma generator and dataset sampling."""
    dataset = SyntheticPlasma4DDataset(
        num_points=1000,
        x_bounds=(-300.0, 300.0),
        y_bounds=(-300.0, 300.0),
        z_bounds=(100.0, 500.0),
        t_bounds=(0.0, 300.0),
        v_km_s=1.0,
        seed=123,
    )

    assert len(dataset) == 1000
    sample = dataset[0]

    coords = sample["coords"]
    values = sample["values"]

    assert coords.shape == (1000, 4)
    assert values.shape == (1000, 1)

    assert torch.all(torch.isfinite(coords))
    assert torch.all(torch.isfinite(values))

    # Check physical z bounds
    df = dataset.df
    assert float(df["z_km"].min()) >= 100.0
    assert float(df["z_km"].max()) <= 500.0
    assert float(df["log10_Ne"].min()) > 0.0


def test_4d_siren_model_forward_and_autograd_curvature():
    """Test 4D SIREN model forward pass and PyTorch autograd spatial-temporal curvature loss."""
    set_seed(42)
    model = MLPINR(
        in_features=4,
        out_features=1,
        hidden_features=64,
        hidden_layers=2,
        activation="sine",
    )

    collocation_coords = torch.randn(50, 4, dtype=torch.float32)

    # Forward pass
    pred = model(collocation_coords)
    assert pred.shape == (50, 1)
    assert torch.all(torch.isfinite(pred))

    # 4D curvature loss (f_xx^2 + f_yy^2 + f_zz^2 + f_tt^2)
    l_curv, details = curvature_loss_4d(model, collocation_coords, loss_type="isotropic")

    assert isinstance(l_curv, torch.Tensor)
    assert torch.isfinite(l_curv)
    assert l_curv.item() >= 0.0

    for key in ["curv_xx", "curv_yy", "curv_zz_quadratic", "curv_tt"]:
        assert key in details
        assert torch.isfinite(details[key])
        assert details[key].item() >= 0.0

    # Test backprop on total loss (data + curvature)
    target = torch.zeros(50, 1)
    l_data = torch.nn.functional.mse_loss(pred, target)
    l_total = l_data + l_curv
    l_total.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Gradient non-finite for {name}"


def test_anisotropic_huber_curvature_loss_4d():
    """Test 4D anisotropic Huber curvature loss calculation and backwards compatibility."""
    set_seed(42)
    model = MLPINR(
        in_features=4,
        out_features=1,
        hidden_features=64,
        hidden_layers=2,
        activation="sine",
    )

    collocation_coords = torch.randn(50, 4, dtype=torch.float32)

    # 1. Isotropic loss (default)
    l_curv_iso, details_iso = curvature_loss_4d(model, collocation_coords, loss_type="isotropic")
    assert details_iso["loss_type"] == "isotropic"
    assert torch.isfinite(l_curv_iso)

    # 2. Anisotropic Huber loss
    l_curv_aniso, details_aniso = curvature_loss_4d(
        model=model,
        coords_col=collocation_coords,
        loss_type="anisotropic_huber",
        lambda_xy=1.0,
        lambda_z=0.1,
        lambda_t=1.0,
        huber_delta_z=0.1,
    )
    assert details_aniso["loss_type"] == "anisotropic_huber"
    assert torch.isfinite(l_curv_aniso)
    assert "curv_zz_huber" in details_aniso
    assert "curv_xy_mixed" in details_aniso

    # Test backprop on anisotropic loss
    pred = model(collocation_coords)
    target = torch.zeros(50, 1)
    l_data = torch.nn.functional.mse_loss(pred, target)
    l_total = l_data + l_curv_aniso
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    l_total.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Gradient non-finite for {name}"


def test_synthetic_4d_training_step_sanity():
    """Sanity check ensuring 4D synthetic training step decreases loss and errors remain finite."""
    set_seed(42)
    dataset = SyntheticPlasma4DDataset(num_points=500, seed=42)
    sample = dataset[0]

    coords = sample["coords"]
    values = sample["values"]

    model = MLPINR(
        in_features=4,
        out_features=1,
        hidden_features=64,
        hidden_layers=2,
        activation="sine",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_curv = 1e-4

    losses = []
    for step in range(25):
        optimizer.zero_grad()
        pred = model(coords)
        l_data = torch.nn.functional.mse_loss(pred, values)
        l_curv, _ = curvature_loss_4d(
            model=model,
            coords_col=coords,
            loss_type="anisotropic_huber",
            lambda_z=0.1,
        )

        l_total = l_data + lambda_curv * l_curv
        assert torch.isfinite(l_total), f"Step {step}: Loss is non-finite!"

        l_total.backward()
        optimizer.step()

        losses.append(l_total.item())

    # Verify loss decreased from start to end
    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]}, final={losses[-1]}"
    assert np.all(np.isfinite(losses))

