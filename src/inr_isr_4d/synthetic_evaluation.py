"""Independent synthetic truth and analytic-derivative evaluation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import atomic_json_save
from .data import AffineScaler, FieldBundle4D
from .evaluation import load_trained_model, point_metrics
from .synthetic import (
    SyntheticFieldConfig,
    SyntheticObservationConfig,
    evaluate_observation_target,
    independent_truth_points,
)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def predict_log_and_physical_derivatives(
    model: torch.nn.Module,
    coordinates: np.ndarray,
    coordinate_scaler: AffineScaler,
    target_scaler: AffineScaler,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = coordinate_scaler.transform(coordinates)
    predictions = []
    derivatives = []
    target_span = float(target_scaler.maximum[0] - target_scaler.minimum[0])
    coordinate_spans = coordinate_scaler.maximum - coordinate_scaler.minimum
    chain = torch.as_tensor(target_span / coordinate_spans, dtype=dtype, device=device)
    model.eval()
    for begin in range(0, len(normalized), chunk_size):
        points = torch.as_tensor(
            normalized[begin : begin + chunk_size], dtype=dtype, device=device
        ).requires_grad_(True)
        prediction_normalized = model(points)
        gradient = torch.autograd.grad(prediction_normalized.sum(), points)[0]
        predictions.append(
            target_scaler.inverse_transform(prediction_normalized.detach().cpu().numpy())
        )
        derivatives.append((gradient * chain).detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(derivatives)


def _relative_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    relative = (prediction - truth) / np.maximum(np.abs(truth), 1.0)
    absolute = np.abs(relative)
    return {
        "mean_relative_error": float(relative.mean()),
        "median_absolute_relative_error": float(np.quantile(absolute, 0.5)),
        "p90_absolute_relative_error": float(np.quantile(absolute, 0.9)),
        "p95_absolute_relative_error": float(np.quantile(absolute, 0.95)),
    }


def evaluate_independent_synthetic_truth(
    *,
    bundle: FieldBundle4D,
    field_config: SyntheticFieldConfig,
    observation_config: SyntheticObservationConfig,
    coordinate_scaler: AffineScaler,
    target_scaler: AffineScaler,
    checkpoint_path: Path,
    output_directory: Path,
    truth_count: int,
    truth_seed: int,
) -> dict[str, Any]:
    """Evaluate independent points never used for fitting or collocation."""

    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Synthetic truth directory already exists: {output_directory}")
    bounds = np.column_stack([bundle.coordinates.min(axis=0), bundle.coordinates.max(axis=0)])
    coordinates, _ = independent_truth_points(
        field_config, count=truth_count, bounds=bounds, seed=truth_seed
    )
    truth = evaluate_observation_target(
        coordinates,
        field_config,
        mode=observation_config.mode,
        integration_duration_sec=observation_config.integration_duration_sec,
        integration_samples=observation_config.integration_samples,
    )
    model, config, _, device, dtype = load_trained_model(checkpoint_path)
    prediction_log, prediction_derivatives = predict_log_and_physical_derivatives(
        model,
        coordinates,
        coordinate_scaler,
        target_scaler,
        device=device,
        dtype=dtype,
        chunk_size=config.runtime.inference_chunk_size,
    )
    truth_log = truth["log10_Ne"][:, None]
    prediction_linear = np.power(10.0, prediction_log)
    truth_linear = truth["Ne"][:, None]
    derivative_metrics = {}
    truth_derivatives = []
    for axis, name in enumerate(("x", "y", "z", "t")):
        values = truth[f"dlog10Ne_d{name}"][:, None]
        truth_derivatives.append(values)
        derivative_metrics[name] = point_metrics(
            prediction_derivatives[:, axis : axis + 1], values
        )
        if not np.isfinite(derivative_metrics[name]["r_squared"]):
            derivative_metrics[name]["r_squared"] = None
    log_metrics = point_metrics(prediction_log, truth_log)
    linear_metrics = point_metrics(prediction_linear, truth_linear)
    for metrics in (log_metrics, linear_metrics):
        if not np.isfinite(metrics["r_squared"]):
            metrics["r_squared"] = None
    summary = {
        "schema_version": 1,
        "status": "complete",
        "truth_count": truth_count,
        "truth_seed": truth_seed,
        "truth_sampling": "independent uniform physical-coordinate points over full observation bounds",
        "observation_target": (
            "integration-averaged linear Ne converted to log10"
            if observation_config.mode == "integration_averaged"
            else "instantaneous log10 Ne"
        ),
        "metrics_log10_density": log_metrics,
        "metrics_linear_density": linear_metrics,
        "relative_linear_density": _relative_metrics(prediction_linear, truth_linear),
        "analytic_first_derivative_metrics": derivative_metrics,
        "midpoint_truth_role": "separate temporal-smearing reference; not substituted for the integration-product target",
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    atomic_json_save(summary, output_directory / "summary.json")
    _atomic_npz(
        output_directory / "predictions.npz",
        coordinates=coordinates,
        truth_log10_ne=truth_log,
        prediction_log10_ne=prediction_log,
        truth_ne_m3=truth_linear,
        prediction_ne_m3=prediction_linear,
        truth_log10_derivatives=np.concatenate(truth_derivatives, axis=1),
        prediction_log10_derivatives=prediction_derivatives,
    )
    return summary
