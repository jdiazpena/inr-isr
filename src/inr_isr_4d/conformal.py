"""Leakage-safe split conformal calibration and grouped empirical evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import math
import numpy as np


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def absolute_residual_scores(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions = _finite_array(predictions, "predictions")
    targets = _finite_array(targets, "targets")
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have identical shapes.")
    return np.abs(targets - predictions).reshape(-1)


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Return strict upper order statistic and its one-indexed clipped rank."""

    scores = _finite_array(scores, "calibration scores").reshape(-1)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one.")
    rank = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(scores))
    quantile = float(np.partition(scores, rank - 1)[rank - 1])
    return quantile, rank


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    quantile: float
    rank: int
    scores: np.ndarray
    calibration_groups: tuple[object, ...]
    calibration_unit: str
    model_identity: str
    prediction_transform: str

    def __post_init__(self) -> None:
        scores = _finite_array(self.scores, "calibration scores").reshape(-1)
        if self.rank < 1 or self.rank > len(scores):
            raise ValueError("Calibration rank is outside the score array.")
        if not math.isfinite(self.quantile) or self.quantile < 0:
            raise ValueError("Calibration quantile must be finite and non-negative.")
        object.__setattr__(self, "scores", scores)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "alpha": self.alpha,
            "quantile": self.quantile,
            "rank_one_indexed": self.rank,
            "scores": self.scores.tolist(),
            "calibration_groups": list(self.calibration_groups),
            "calibration_unit": self.calibration_unit,
            "model_identity": self.model_identity,
            "prediction_transform": self.prediction_transform,
            "score_convention": "absolute_residual_symmetric",
        }


def calibration_from_state(state: Mapping[str, object]) -> ConformalCalibration:
    required = {
        "schema_version",
        "alpha",
        "quantile",
        "rank_one_indexed",
        "scores",
        "calibration_groups",
        "calibration_unit",
        "model_identity",
        "prediction_transform",
        "score_convention",
    }
    if set(state) != required or state["schema_version"] != 1:
        raise ValueError("Invalid conformal calibration artifact schema.")
    if state["score_convention"] != "absolute_residual_symmetric":
        raise ValueError("Unsupported conformal score convention.")
    return ConformalCalibration(
        alpha=float(state["alpha"]),
        quantile=float(state["quantile"]),
        rank=int(state["rank_one_indexed"]),
        scores=np.asarray(state["scores"], dtype=float),
        calibration_groups=tuple(state["calibration_groups"]),
        calibration_unit=str(state["calibration_unit"]),
        model_identity=str(state["model_identity"]),
        prediction_transform=str(state["prediction_transform"]),
    )


def calibrate_split_conformal(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    calibration_groups: tuple[object, ...],
    calibration_unit: str,
    model_identity: str,
    prediction_transform: str,
) -> ConformalCalibration:
    scores = absolute_residual_scores(predictions, targets)
    quantile, rank = finite_sample_quantile(scores, alpha)
    if not calibration_groups:
        raise ValueError("Calibration groups cannot be empty.")
    if not model_identity or not prediction_transform or not calibration_unit:
        raise ValueError("Calibration provenance strings cannot be empty.")
    return ConformalCalibration(
        alpha=alpha,
        quantile=quantile,
        rank=rank,
        scores=scores,
        calibration_groups=calibration_groups,
        calibration_unit=calibration_unit,
        model_identity=model_identity,
        prediction_transform=prediction_transform,
    )


def conformal_intervals(
    predictions: np.ndarray, calibration: ConformalCalibration
) -> tuple[np.ndarray, np.ndarray]:
    predictions = _finite_array(predictions, "predictions")
    return predictions - calibration.quantile, predictions + calibration.quantile


def empirical_interval_metrics(
    targets: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float | int]:
    targets = _finite_array(targets, "targets")
    lower = _finite_array(lower, "lower interval")
    upper = _finite_array(upper, "upper interval")
    if targets.shape != lower.shape or targets.shape != upper.shape:
        raise ValueError("Targets and interval arrays must have identical shapes.")
    if np.any(upper < lower):
        raise ValueError("Upper interval bounds cannot be below lower bounds.")
    covered = (targets >= lower) & (targets <= upper)
    widths = upper - lower
    return {
        "count": int(targets.size),
        "empirical_coverage": float(covered.mean()),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
    }


def stratified_interval_metrics(
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    strata: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    size = np.asarray(targets).size
    flat_target = np.asarray(targets).reshape(-1)
    flat_lower = np.asarray(lower).reshape(-1)
    flat_upper = np.asarray(upper).reshape(-1)
    for stratum_name, labels in strata.items():
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != size:
            raise ValueError(f"Stratum {stratum_name} does not match target size.")
        groups: dict[str, dict[str, float | int]] = {}
        for label in np.unique(labels):
            mask = labels == label
            groups[str(label)] = empirical_interval_metrics(
                flat_target[mask], flat_lower[mask], flat_upper[mask]
            )
        result[stratum_name] = groups
    return result


def group_bootstrap_coverage(
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if repetitions <= 0 or seed < 0:
        raise ValueError("Bootstrap repetitions must be positive and seed non-negative.")
    flat_target = _finite_array(targets, "targets").reshape(-1)
    flat_lower = _finite_array(lower, "lower").reshape(-1)
    flat_upper = _finite_array(upper, "upper").reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if not (len(flat_target) == len(flat_lower) == len(flat_upper) == len(groups)):
        raise ValueError("Bootstrap arrays must have equal lengths.")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Group bootstrap requires at least two groups.")
    rng = np.random.default_rng(seed)
    coverages = []
    covered = (flat_target >= flat_lower) & (flat_target <= flat_upper)
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        coverages.append(float(covered[indices].mean()))
    values = np.asarray(coverages)
    return {
        "repetitions": repetitions,
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if repetitions > 1 else 0.0,
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
    }
