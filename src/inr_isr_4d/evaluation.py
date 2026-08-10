"""Checkpoint-consuming prediction, conformal calibration, and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import atomic_json_save
from .config import FourDConfig, config_from_mapping
from .conformal import (
    calibration_from_state,
    calibrate_split_conformal,
    conformal_intervals,
    empirical_interval_metrics,
    group_bootstrap_coverage,
    stratified_interval_metrics,
)
from .data import AffineScaler, FieldBundle4D
from .model import SIREN4D
from .splits import GroupSplit4D, observation_group_ids
from .training import resolve_device


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if predictions.shape != targets.shape or len(targets) == 0:
        raise ValueError("Point metric arrays must be non-empty and shape matched.")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise ValueError("Point metric arrays must be finite.")
    residual = predictions - targets
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    return {
        "count": len(targets),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r_squared": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan"),
        "p50_absolute_error": float(np.quantile(np.abs(residual), 0.5)),
        "p90_absolute_error": float(np.quantile(np.abs(residual), 0.9)),
        "p95_absolute_error": float(np.quantile(np.abs(residual), 0.95)),
    }


def predict_chunked(
    model: torch.nn.Module,
    physical_coordinates: np.ndarray,
    coordinate_scaler: AffineScaler,
    target_scaler: AffineScaler,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("Prediction chunk size must be positive.")
    normalized = coordinate_scaler.transform(np.asarray(physical_coordinates))
    predictions = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(normalized), chunk_size):
            coordinates = torch.as_tensor(
                normalized[begin : begin + chunk_size], dtype=dtype, device=device
            )
            predictions.append(model(coordinates).detach().cpu().numpy())
    normalized_prediction = np.concatenate(predictions, axis=0)
    return target_scaler.inverse_transform(normalized_prediction)


def load_trained_model(
    checkpoint_path: Path,
) -> tuple[SIREN4D, FourDConfig, dict[str, Any], torch.device, torch.dtype]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = config_from_mapping(state["resolved_config"])
    device = resolve_device(config.runtime.device)
    dtype = torch.float32 if config.runtime.precision == "float32" else torch.float64
    model = SIREN4D(
        hidden_features=config.model.hidden_features,
        hidden_layers=config.model.hidden_layers,
        activation=config.model.activation,
        first_omega_0=config.model.first_omega_0,
        hidden_omega_0=config.model.hidden_omega_0,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state["model"])
    return model, config, state, device, dtype


def _nearest_support_distance(query: np.ndarray, training: np.ndarray) -> np.ndarray:
    values = []
    for begin in range(0, len(query), 4096):
        difference = query[begin : begin + 4096, None, :3] - training[None, :, :3]
        values.append(np.sqrt(np.sum(difference**2, axis=2)).min(axis=1))
    return np.concatenate(values)


def _bin_labels(values: np.ndarray, prefix: str, bins: int = 4) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        return np.full(len(values), f"{prefix}_all")
    positions = np.digitize(values, edges[1:-1], right=True)
    return np.array([f"{prefix}_{position}" for position in positions])


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    split: GroupSplit4D,
    coordinate_scaler: AffineScaler,
    target_scaler: AffineScaler,
) -> None:
    if checkpoint["splits"] != split.state_dict():
        raise ValueError("The checkpoint split does not match the evaluation split.")
    expected_preprocessing = {
        "coordinates": coordinate_scaler.state_dict(),
        "targets": target_scaler.state_dict(),
    }
    if checkpoint["preprocessing"] != expected_preprocessing:
        raise ValueError("The checkpoint preprocessing does not match evaluation preprocessing.")


def calibrate_checkpoint(
    *,
    bundle: FieldBundle4D,
    split: GroupSplit4D,
    coordinate_scaler: AffineScaler,
    target_scaler: AffineScaler,
    checkpoint_path: Path,
    calibration_path: Path,
    alpha: float,
) -> dict[str, object]:
    """Run only conformal calibration; do not inspect test targets."""

    calibration_path = Path(calibration_path)
    if calibration_path.exists():
        raise FileExistsError(f"Calibration artifact already exists: {calibration_path}")
    if len(split.calibration_indices) == 0:
        raise ValueError("Conformal calibration split is empty.")
    model, config, checkpoint, device, dtype = load_trained_model(checkpoint_path)
    _validate_checkpoint_contract(checkpoint, split, coordinate_scaler, target_scaler)
    calibration_bundle = bundle.subset(split.calibration_indices)
    predictions = predict_chunked(
        model,
        calibration_bundle.coordinates,
        coordinate_scaler,
        target_scaler,
        device=device,
        dtype=dtype,
        chunk_size=config.runtime.inference_chunk_size,
    )
    calibration = calibrate_split_conformal(
        predictions,
        calibration_bundle.targets,
        alpha=alpha,
        calibration_groups=split.calibration_groups,
        calibration_unit=split.unit,
        model_identity="sha256:" + file_sha256(checkpoint_path),
        prediction_transform="inverse training-only affine transform in log10 electron density",
    )
    state = calibration.state_dict()
    atomic_json_save(state, calibration_path)
    return state


def evaluate_checkpoint(
    *,
    bundle: FieldBundle4D,
    split: GroupSplit4D,
    coordinate_scaler: AffineScaler,
    target_scaler: AffineScaler,
    checkpoint_path: Path,
    calibration_path: Path,
    output_directory: Path,
    bootstrap_repetitions: int = 200,
) -> dict[str, Any]:
    """Evaluate untouched test groups using an already saved calibration artifact."""

    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output_directory}")
    if len(split.test_indices) == 0:
        raise ValueError("Conformal test split is empty.")
    model, config, checkpoint, device, dtype = load_trained_model(checkpoint_path)
    _validate_checkpoint_contract(checkpoint, split, coordinate_scaler, target_scaler)
    with Path(calibration_path).open("r", encoding="utf-8") as handle:
        calibration = calibration_from_state(json.load(handle))
    model_identity = "sha256:" + file_sha256(checkpoint_path)
    if calibration.model_identity != model_identity:
        raise ValueError("Calibration artifact belongs to a different model checkpoint.")
    if calibration.calibration_unit != split.unit or tuple(calibration.calibration_groups) != tuple(split.calibration_groups):
        raise ValueError("Calibration groups/unit differ from the evaluation split.")
    output_directory.mkdir(parents=True, exist_ok=False)

    test_bundle = bundle.subset(split.test_indices)
    test_prediction = predict_chunked(
        model,
        test_bundle.coordinates,
        coordinate_scaler,
        target_scaler,
        device=device,
        dtype=dtype,
        chunk_size=config.runtime.inference_chunk_size,
    )
    lower, upper = conformal_intervals(test_prediction, calibration)
    support_distance = _nearest_support_distance(
        test_bundle.coordinates, bundle.coordinates[split.train_indices]
    )
    test_unit_ids = observation_group_ids(test_bundle, split.unit)
    strata = {
        "beam": test_bundle.beam_ids,
        "time_block": test_bundle.time_ids,
        "altitude": _bin_labels(test_bundle.coordinates[:, 2], "altitude"),
        "support_distance": _bin_labels(support_distance, "support_distance"),
    }
    if len(np.unique(test_unit_ids)) >= 2:
        bootstrap: dict[str, Any] = group_bootstrap_coverage(
            test_bundle.targets,
            lower,
            upper,
            test_unit_ids,
            repetitions=bootstrap_repetitions,
            seed=split.seed,
        )
    else:
        bootstrap = {
            "status": "unavailable",
            "reason": "At least two test groups are required for a group bootstrap.",
        }
    point = point_metrics(test_prediction, test_bundle.targets)
    if not np.isfinite(point["r_squared"]):
        point["r_squared"] = None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_identity": model_identity,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "split": split.state_dict(),
        "point_metrics_log10_density": point,
        "marginal_interval_metrics": empirical_interval_metrics(test_bundle.targets, lower, upper),
        "stratified_interval_metrics": stratified_interval_metrics(test_bundle.targets, lower, upper, strata),
        "group_bootstrap": bootstrap,
        "interpretation": {
            "coverage": "empirical point-level coverage on untouched held-out groups",
            "calibration_unit": split.unit,
            "test_unit": split.unit,
            "exchangeability_assumption": "Calibration and test scores must be exchangeable at the declared sampling unit.",
            "limitation": "Correlated and grouped radar sampling can violate point-level exchangeability; no unconditional coverage guarantee is claimed.",
        },
    }
    atomic_json_save(summary, output_directory / "evaluation_summary.json")
    _atomic_npz(
        output_directory / "predictions.npz",
        coordinates=test_bundle.coordinates,
        targets=test_bundle.targets,
        predictions=test_prediction,
        residuals=test_prediction - test_bundle.targets,
        interval_lower=lower,
        interval_upper=upper,
        beam_ids=test_bundle.beam_ids,
        time_ids=test_bundle.time_ids,
        group_ids=test_bundle.group_ids,
        support_distance_km=support_distance,
    )
    atomic_json_save(
        {
            "schema_version": 1,
            "status": "complete",
            "model_identity": model_identity,
            "summary": "evaluation_summary.json",
            "predictions": "predictions.npz",
        },
        output_directory / "COMPLETED.json",
    )
    return summary


def calibrate_and_evaluate(
    *,
    bundle: FieldBundle4D,
    split: GroupSplit4D,
    coordinate_scaler: AffineScaler,
    target_scaler: AffineScaler,
    checkpoint_path: Path,
    output_directory: Path,
    alpha: float,
    bootstrap_repetitions: int = 200,
) -> dict[str, Any]:
    """Calibrate on saved calibration groups, then evaluate untouched test groups once."""

    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty: {output_directory}")
    if len(split.calibration_indices) == 0 or len(split.test_indices) == 0:
        raise ValueError("Conformal calibration and test splits must both be non-empty.")
    model, config, checkpoint, device, dtype = load_trained_model(checkpoint_path)
    if checkpoint["splits"] != split.state_dict():
        raise ValueError("The checkpoint split does not match the evaluation split.")
    expected_preprocessing = {
        "coordinates": coordinate_scaler.state_dict(),
        "targets": target_scaler.state_dict(),
    }
    if checkpoint["preprocessing"] != expected_preprocessing:
        raise ValueError("The checkpoint preprocessing does not match evaluation preprocessing.")
    output_directory.mkdir(parents=True, exist_ok=False)

    calibration_bundle = bundle.subset(split.calibration_indices)
    test_bundle = bundle.subset(split.test_indices)
    calibration_prediction = predict_chunked(
        model,
        calibration_bundle.coordinates,
        coordinate_scaler,
        target_scaler,
        device=device,
        dtype=dtype,
        chunk_size=config.runtime.inference_chunk_size,
    )
    model_identity = "sha256:" + file_sha256(checkpoint_path)
    calibration = calibrate_split_conformal(
        calibration_prediction,
        calibration_bundle.targets,
        alpha=alpha,
        calibration_groups=split.calibration_groups,
        calibration_unit=split.unit,
        model_identity=model_identity,
        prediction_transform="inverse training-only affine transform in log10 electron density",
    )
    atomic_json_save(calibration.state_dict(), output_directory / "calibration.json")

    test_prediction = predict_chunked(
        model,
        test_bundle.coordinates,
        coordinate_scaler,
        target_scaler,
        device=device,
        dtype=dtype,
        chunk_size=config.runtime.inference_chunk_size,
    )
    lower, upper = conformal_intervals(test_prediction, calibration)
    support_distance = _nearest_support_distance(
        test_bundle.coordinates, bundle.coordinates[split.train_indices]
    )
    test_unit_ids = observation_group_ids(test_bundle, split.unit)
    strata = {
        "beam": test_bundle.beam_ids,
        "time_block": test_bundle.time_ids,
        "altitude": _bin_labels(test_bundle.coordinates[:, 2], "altitude"),
        "support_distance": _bin_labels(support_distance, "support_distance"),
    }
    bootstrap: dict[str, Any]
    if len(np.unique(test_unit_ids)) >= 2:
        bootstrap = group_bootstrap_coverage(
            test_bundle.targets,
            lower,
            upper,
            test_unit_ids,
            repetitions=bootstrap_repetitions,
            seed=split.seed,
        )
    else:
        bootstrap = {
            "status": "unavailable",
            "reason": "At least two test groups are required for a group bootstrap.",
        }

    point = point_metrics(test_prediction, test_bundle.targets)
    if not np.isfinite(point["r_squared"]):
        point["r_squared"] = None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_identity": model_identity,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "split": split.state_dict(),
        "point_metrics_log10_density": point,
        "marginal_interval_metrics": empirical_interval_metrics(test_bundle.targets, lower, upper),
        "stratified_interval_metrics": stratified_interval_metrics(
            test_bundle.targets, lower, upper, strata
        ),
        "group_bootstrap": bootstrap,
        "interpretation": {
            "coverage": "empirical point-level coverage on untouched held-out groups",
            "calibration_unit": split.unit,
            "test_unit": split.unit,
            "exchangeability_assumption": "Calibration and test scores must be exchangeable at the declared sampling unit.",
            "limitation": "Correlated and grouped radar sampling can violate point-level exchangeability; no unconditional coverage guarantee is claimed.",
        },
    }
    atomic_json_save(summary, output_directory / "evaluation_summary.json")
    _atomic_npz(
        output_directory / "predictions.npz",
        coordinates=test_bundle.coordinates,
        targets=test_bundle.targets,
        predictions=test_prediction,
        residuals=test_prediction - test_bundle.targets,
        interval_lower=lower,
        interval_upper=upper,
        beam_ids=test_bundle.beam_ids,
        time_ids=test_bundle.time_ids,
        group_ids=test_bundle.group_ids,
        support_distance_km=support_distance,
    )
    atomic_json_save(
        {
            "schema_version": 1,
            "status": "complete",
            "model_identity": model_identity,
            "summary": "evaluation_summary.json",
            "predictions": "predictions.npz",
        },
        output_directory / "COMPLETED.json",
    )
    return summary
