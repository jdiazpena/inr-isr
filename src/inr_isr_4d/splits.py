"""Leakage-safe group splits constructed before target preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import AffineScaler, FieldBundle4D
from .training import TrainingProblem4D


GroupUnit = Literal["beam", "time_block", "beam_time"]
SplitStrategy = Literal["random", "clustered"]


def _python(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def observation_group_ids(bundle: FieldBundle4D, unit: GroupUnit) -> np.ndarray:
    if unit == "beam":
        return bundle.beam_ids.copy()
    if unit == "time_block":
        return bundle.time_ids.copy()
    if unit == "beam_time":
        return np.array(
            [f"{_python(beam)}::{_python(time)}" for beam, time in zip(bundle.beam_ids, bundle.time_ids)]
        )
    raise ValueError(f"Unknown group unit: {unit}")


@dataclass(frozen=True)
class GroupSplit4D:
    unit: GroupUnit
    strategy: SplitStrategy
    seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: tuple[object, ...]
    validation_groups: tuple[object, ...]
    calibration_groups: tuple[object, ...]
    test_groups: tuple[object, ...]

    def __post_init__(self) -> None:
        arrays = []
        for name in (
            "train_indices",
            "validation_indices",
            "calibration_indices",
            "test_indices",
        ):
            array = np.asarray(getattr(self, name), dtype=np.int64)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one dimensional.")
            object.__setattr__(self, name, array)
            arrays.append(array)
        nonempty = [array for array in arrays if len(array)]
        combined = np.concatenate(nonempty) if nonempty else np.array([], dtype=np.int64)
        if len(np.unique(combined)) != len(combined):
            raise ValueError("Split indices overlap.")
        group_sets = [set(getattr(self, name)) for name in (
            "train_groups", "validation_groups", "calibration_groups", "test_groups"
        )]
        for i, left in enumerate(group_sets):
            for right in group_sets[i + 1 :]:
                if left & right:
                    raise ValueError("Split groups overlap.")

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "unit": self.unit,
            "strategy": self.strategy,
            "seed": self.seed,
            "indices": {
                "train": self.train_indices.tolist(),
                "validation": self.validation_indices.tolist(),
                "calibration": self.calibration_indices.tolist(),
                "test": self.test_indices.tolist(),
            },
            "groups": {
                "train": [_python(value) for value in self.train_groups],
                "validation": [_python(value) for value in self.validation_groups],
                "calibration": [_python(value) for value in self.calibration_groups],
                "test": [_python(value) for value in self.test_groups],
            },
            "counts": {
                "observations": {
                    "train": len(self.train_indices),
                    "validation": len(self.validation_indices),
                    "calibration": len(self.calibration_indices),
                    "test": len(self.test_indices),
                },
                "groups": {
                    "train": len(self.train_groups),
                    "validation": len(self.validation_groups),
                    "calibration": len(self.calibration_groups),
                    "test": len(self.test_groups),
                },
            },
        }


def _centroids(bundle: FieldBundle4D, ids: np.ndarray, groups: np.ndarray) -> np.ndarray:
    rows = []
    for group in groups:
        coordinates = bundle.coordinates[ids == group]
        rows.append(coordinates.mean(axis=0))
    centers = np.asarray(rows, dtype=float)
    spans = np.ptp(centers, axis=0)
    spans[spans == 0] = 1.0
    return (centers - centers.min(axis=0)) / spans


def _cluster_order(
    candidates: np.ndarray,
    centers: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.array([], dtype=candidates.dtype)
    center_position = int(rng.integers(len(candidates)))
    distances = np.linalg.norm(centers - centers[center_position], axis=1)
    order = np.argsort(distances, kind="stable")
    return candidates[order[:count]]


def make_group_split(
    bundle: FieldBundle4D,
    *,
    unit: GroupUnit,
    strategy: SplitStrategy,
    validation_group_count: int,
    calibration_group_count: int,
    test_group_count: int,
    seed: int,
) -> GroupSplit4D:
    """Withhold exact group counts without radius-based count expansion."""

    if min(validation_group_count, calibration_group_count, test_group_count, seed) < 0:
        raise ValueError("Group counts and seed must be non-negative.")
    ids = observation_group_ids(bundle, unit)
    groups = np.unique(ids)
    held_count = validation_group_count + calibration_group_count + test_group_count
    if held_count >= len(groups):
        raise ValueError("The requested groups leave no training group.")
    rng = np.random.default_rng(seed)

    if strategy == "random":
        ordered = groups[rng.permutation(len(groups))]
        cursor = 0
        test_groups = ordered[cursor : cursor + test_group_count]
        cursor += test_group_count
        calibration_groups = ordered[cursor : cursor + calibration_group_count]
        cursor += calibration_group_count
        validation_groups = ordered[cursor : cursor + validation_group_count]
    elif strategy == "clustered":
        centers = _centroids(bundle, ids, groups)
        test_groups = _cluster_order(groups, centers, test_group_count, rng)
        remaining_mask = ~np.isin(groups, test_groups)
        remaining = groups[remaining_mask]
        remaining_centers = centers[remaining_mask]
        calibration_groups = _cluster_order(
            remaining, remaining_centers, calibration_group_count, rng
        )
        remaining_mask_2 = ~np.isin(remaining, calibration_groups)
        validation_groups = _cluster_order(
            remaining[remaining_mask_2],
            remaining_centers[remaining_mask_2],
            validation_group_count,
            rng,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    held = np.concatenate([test_groups, calibration_groups, validation_groups])
    train_groups = groups[~np.isin(groups, held)]

    def indices(selected: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(ids, selected))

    return GroupSplit4D(
        unit=unit,
        strategy=strategy,
        seed=seed,
        train_indices=indices(train_groups),
        validation_indices=indices(validation_groups),
        calibration_indices=indices(calibration_groups),
        test_indices=indices(test_groups),
        train_groups=tuple(_python(value) for value in train_groups),
        validation_groups=tuple(_python(value) for value in validation_groups),
        calibration_groups=tuple(_python(value) for value in calibration_groups),
        test_groups=tuple(_python(value) for value in test_groups),
    )


def prepare_training_problem(bundle: FieldBundle4D, split: GroupSplit4D) -> TrainingProblem4D:
    """Fit all learned preprocessing exclusively on saved training observations."""

    training = bundle.subset(split.train_indices)
    return TrainingProblem4D(
        bundle=bundle,
        train_indices=split.train_indices,
        coordinate_scaler=AffineScaler.fit(training.coordinates),
        target_scaler=AffineScaler.fit(training.targets),
        split_state=split.state_dict(),
    )
