"""Mandatory CUDA tests: missing access is a failure, never a passing skip."""

from __future__ import annotations

import torch

from inr_isr.device import resolve_device
from inr_isr.model import build_model
from inr_isr.training.derivatives import coordinate_gradient, hessian_rows
from inr_isr.training.state import capture_rng_state, restore_rng_state
from inr_isr.training.steps import data_objective, parameter_update
from inr_isr_4d.regularization import second_derivatives_4d


def test_real_model_batch_gradient_and_update_are_on_cuda():
    device = resolve_device()
    assert device.type == "cuda"
    model = build_model("xyt", hidden_features=8, hidden_layers=1,
                        activation="sine", first_omega_0=5.0, hidden_omega_0=5.0).to(device)
    points = torch.tensor([[0.0, 0.1, 0.2], [0.2, -0.1, 0.3]], device=device)
    target = torch.ones((2, 1), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    _, loss = data_objective(model, points, target, recipe="xyt")
    parameter_update(optimizer, recipe="xyt", total_loss=loss)
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    assert points.device.type == "cuda" and target.device.type == "cuda"
    assert all(parameter.grad is not None and parameter.grad.device.type == "cuda"
               for parameter in model.parameters())
    torch.cuda.synchronize()
    assert any(not torch.equal(prior, parameter) for prior, parameter in zip(before, model.parameters()))


def test_native_xyt_and_xyzt_second_derivatives_are_on_cuda():
    device = resolve_device()
    xyt = build_model("xyt", hidden_features=8, hidden_layers=1,
                      activation="sine", first_omega_0=5.0, hidden_omega_0=5.0).to(device)
    points3 = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                           device=device, requires_grad=True)
    first = coordinate_gradient(xyt(points3), points3, only_inputs=True)
    rows = hessian_rows(first, points3, axes=("x", "y", "t"),
                        selected=("x", "y", "t"), only_inputs=True)
    assert first.device.type == "cuda"
    assert all(row.device.type == "cuda" for row in rows.values())

    xyzt = build_model("xyzt", hidden_features=8, hidden_layers=1,
                       activation="sine", first_omega_0=5.0, hidden_omega_0=5.0).to(device)
    points4 = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], device=device)
    derivatives = second_derivatives_4d(xyzt, points4)
    assert derivatives
    assert all(value.device.type == "cuda" for value in derivatives.values())
    torch.cuda.synchronize()


def test_rng_state_loaded_onto_cuda_restores_cpu_and_cuda_generators(tmp_path):
    """CUDA map-location must not make the CPU RNG state unrestorable."""
    resolve_device()
    checkpoint = tmp_path / "rng.pt"
    torch.save({"rng": capture_rng_state()}, checkpoint)
    transferred = torch.load(checkpoint, map_location="cuda", weights_only=False)["rng"]
    assert transferred["torch_cpu"].device.type == "cuda"
    assert transferred["torch_cuda"]
    assert all(item.device.type == "cuda" for item in transferred["torch_cuda"])

    torch.manual_seed(90210)
    torch.cuda.manual_seed_all(90210)
    restore_rng_state(transferred)

    assert torch.equal(torch.get_rng_state(), transferred["torch_cpu"].cpu())
    assert all(
        torch.equal(actual, expected.cpu())
        for actual, expected in zip(torch.cuda.get_rng_state_all(), transferred["torch_cuda"])
    )
