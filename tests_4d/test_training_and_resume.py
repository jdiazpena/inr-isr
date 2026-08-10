"""Canonical 4D training, microbatch, controller, and resume tests."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inr_isr_4d.config import (
    CollocationConfig,
    DerivativePriorConfig,
    FourDConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
)
from inr_isr_4d.controller import ReferenceRatioController
from inr_isr_4d.constraints import ConstraintEvaluation
from inr_isr_4d.data import AffineScaler, FieldBundle4D
from inr_isr_4d.model import SIREN4D
from inr_isr_4d.regularization import derivative_prior_4d
from inr_isr_4d.training import TrainingProblem4D, resolve_device, train_4d


class EngineeringOnlyConstraint:
    """Engineering-only fake used to prove the future constraint interface."""

    name = "test_only_output_energy"
    version = "1"

    def evaluate(self, model, normalized_coordinates, context):
        assert context["coordinate_order"] == ("x", "y", "z", "t")
        loss = 1.0e-7 * torch.mean(model(normalized_coordinates) ** 2)
        return ConstraintEvaluation(loss=loss, diagnostics={"test_scale": 1.0e-7})


def make_problem() -> TrainingProblem4D:
    generator = np.random.default_rng(4)
    coordinates = generator.uniform(size=(32, 4))
    coordinates[:, 0] *= 100.0
    coordinates[:, 1] *= 120.0
    coordinates[:, 2] = 200.0 + coordinates[:, 2] * 300.0
    coordinates[:, 3] *= 600.0
    targets = (
        10.0
        + 0.1 * np.sin(coordinates[:, 0:1] / 30.0)
        + 0.2 * coordinates[:, 3:4] / 600.0
    )
    bundle = FieldBundle4D(
        coordinates=coordinates,
        targets=targets,
        beam_ids=np.arange(32) % 8,
        time_ids=np.arange(32) // 8,
        group_ids=np.array([f"{i % 8}:{i // 8}" for i in range(32)]),
    )
    train_indices = np.arange(24)
    return TrainingProblem4D(
        bundle=bundle,
        train_indices=train_indices,
        coordinate_scaler=AffineScaler.fit(coordinates[train_indices]),
        target_scaler=AffineScaler.fit(targets[train_indices]),
        split_state={"train": train_indices.tolist(), "calibration": [24, 25, 26, 27], "test": [28, 29, 30, 31]},
    )


def make_config(*, num_steps: int = 6, derivative: bool = True) -> FourDConfig:
    return FourDConfig(
        model=ModelConfig(hidden_features=12, hidden_layers=1),
        optimization=OptimizationConfig(
            learning_rate=2.0e-4, num_steps=num_steps, data_batch_size=11, seed=0
        ),
        collocation=CollocationConfig(
            mode="sobol",
            pool_size=17,
            batch_size=13,
            derivative_microbatch_size=5,
            resample_every=2,
            seed=3,
        ),
        derivative_prior=DerivativePriorConfig(
            mode="legacy_diagonal_4d" if derivative else "none",
            weight=1.0e-5 if derivative else 0.0,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            precision="float32",
            inference_chunk_size=9,
            diagnostic_probe_size=7,
            history_every=2,
            checkpoint_every=2,
        ),
    ).validate()


def test_requested_unavailable_cuda_is_not_silently_changed() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_device("cuda")


def test_reference_ratio_controller_round_trip_and_update() -> None:
    config = DerivativePriorConfig(
        mode="legacy_diagonal_4d",
        weight=1.0,
        adaptive=True,
        warmup_steps=0,
        ramp_steps=0,
        update_every=1,
        smoothing=1.0,
        target_horizontal_ratio=0.5,
        target_vertical_ratio=0.25,
        target_temporal_ratio=0.1,
    )
    controller = ReferenceRatioController(config)
    weights = controller.update(
        0, data_loss=2.0, components={"horizontal": 4.0, "vertical": 1.0, "temporal": 0.5}
    )
    assert weights == pytest.approx({"horizontal": 0.25, "vertical": 0.5, "temporal": 0.4})
    restored = ReferenceRatioController(config)
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()


def test_chunked_and_unchunked_derivative_reductions_match() -> None:
    torch.manual_seed(8)
    whole_model = SIREN4D(hidden_features=8, hidden_layers=1).double()
    chunk_model = SIREN4D(hidden_features=8, hidden_layers=1).double()
    chunk_model.load_state_dict(whole_model.state_dict())
    coordinates = torch.randn(11, 4, dtype=torch.float64)
    whole, _ = derivative_prior_4d(
        whole_model, coordinates, mode="anisotropic_huber_4d", weight_vertical=0.2
    )
    whole.backward()
    chunk_total = 0.0
    for points in coordinates.split([4, 4, 3]):
        value, _ = derivative_prior_4d(
            chunk_model, points, mode="anisotropic_huber_4d", weight_vertical=0.2
        )
        fraction = len(points) / len(coordinates)
        (fraction * value).backward()
        chunk_total += fraction * value.detach()
    torch.testing.assert_close(chunk_total, whole.detach(), rtol=1e-12, atol=1e-12)
    for whole_parameter, chunk_parameter in zip(whole_model.parameters(), chunk_model.parameters()):
        torch.testing.assert_close(whole_parameter.grad, chunk_parameter.grad, rtol=1e-10, atol=1e-10)


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    problem = make_problem()
    config = make_config()
    uninterrupted = train_4d(problem, config, tmp_path / "uninterrupted")
    interrupted = train_4d(
        problem, config, tmp_path / "resumed", stop_after_steps=3
    )
    assert interrupted.steps_completed == 3 and not interrupted.complete
    resumed = train_4d(problem, config, tmp_path / "resumed", resume=True)
    assert uninterrupted.complete and resumed.complete
    for expected, actual in zip(uninterrupted.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert uninterrupted.best_data_loss == resumed.best_data_loss
    checkpoint = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
    assert checkpoint["next_step"] == 6
    assert checkpoint["scheduler"] is None
    assert set(checkpoint) >= {
        "model", "optimizer", "preprocessing", "splits", "adaptive_controller", "rng"
    }
    completion = json.loads((tmp_path / "resumed" / "COMPLETED.json").read_text())
    assert completion["status"] == "complete" and completion["steps_completed"] == 6
    records = [
        json.loads(line)
        for line in (tmp_path / "resumed" / "history.jsonl").read_text().splitlines()
    ]
    assert all(record["timestamp_utc"].endswith("+00:00") for record in records)
    uninterrupted_records = [
        json.loads(line)
        for line in (tmp_path / "uninterrupted" / "history.jsonl").read_text().splitlines()
    ]
    assert records[-1]["diagnostic_probe"] == uninterrupted_records[-1]["diagnostic_probe"]
    assert records[-1]["derivative_health"] == uninterrupted_records[-1]["derivative_health"]
    assert all(
        set(record) >= {
            "base_component_weights",
            "target_reference_ratios",
            "observed_weighted_to_data_ratios",
        }
        for record in records
    )


def test_collision_refusal_preserves_existing_run(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("historical")
    with pytest.raises(FileExistsError):
        train_4d(make_problem(), make_config(num_steps=1, derivative=False), output)
    assert sentinel.read_text() == "historical"


def test_future_constraint_interface_is_wired_without_claiming_physics(tmp_path: Path) -> None:
    result = train_4d(
        make_problem(),
        make_config(num_steps=2, derivative=False),
        tmp_path / "constraint",
        additional_constraints=[EngineeringOnlyConstraint()],
    )
    checkpoint = torch.load(result.output_directory / "checkpoint.pt", weights_only=False)
    assert checkpoint["additional_constraints"] == [
        {"name": "test_only_output_energy", "version": "1"}
    ]
    records = [
        json.loads(line)
        for line in (result.output_directory / "history.jsonl").read_text().splitlines()
    ]
    assert records[-1]["additional_constraint_losses"]["test_only_output_energy"] > 0
