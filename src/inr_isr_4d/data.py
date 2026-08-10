"""Explicit immutable-field and conventional sample-dataset contracts for 4D."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


COORDINATE_NAMES = ("x_km", "y_km", "z_km", "t_sec")


def _immutable_array(value: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value).copy()
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FieldBundle4D:
    """A complete, immutable collection of observations and identifiers.

    Coordinate columns are always physical ``(x_km, y_km, z_km, t_sec)``.  This
    object is not a PyTorch Dataset; callers ask for its arrays explicitly.
    """

    coordinates: np.ndarray
    targets: np.ndarray
    beam_ids: np.ndarray
    time_ids: np.ndarray
    group_ids: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = _immutable_array(self.coordinates, ndim=2, name="coordinates")
        targets = _immutable_array(self.targets, ndim=2, name="targets")
        if coordinates.shape[1] != 4:
            raise ValueError("coordinates must have columns (x_km, y_km, z_km, t_sec).")
        if targets.shape[1] < 1:
            raise ValueError("targets must have at least one output column.")
        n = coordinates.shape[0]
        if n == 0:
            raise ValueError("A field bundle cannot be empty.")

        identifiers: dict[str, np.ndarray] = {}
        for name in ("beam_ids", "time_ids", "group_ids"):
            array = np.asarray(getattr(self, name)).copy()
            if array.ndim != 1 or len(array) != n:
                raise ValueError(f"{name} must be a one-dimensional array of length {n}.")
            array.setflags(write=False)
            identifiers[name] = array
        if len(targets) != n:
            raise ValueError("coordinates and targets must contain the same number of rows.")

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "targets", targets)
        for name, array in identifiers.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def size(self) -> int:
        return int(self.coordinates.shape[0])

    def subset(self, indices: np.ndarray) -> "FieldBundle4D":
        index = np.asarray(indices)
        return FieldBundle4D(
            coordinates=self.coordinates[index],
            targets=self.targets[index],
            beam_ids=self.beam_ids[index],
            time_ids=self.time_ids[index],
            group_ids=self.group_ids[index],
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class AffineScaler:
    """Per-column affine map fitted on a declared training subset only."""

    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = _immutable_array(self.minimum, ndim=1, name="minimum")
        maximum = _immutable_array(self.maximum, ndim=1, name="maximum")
        if minimum.shape != maximum.shape:
            raise ValueError("minimum and maximum must have identical shapes.")
        if np.any(maximum <= minimum):
            raise ValueError("Every fitted column must have a positive range.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def fit(cls, training_values: np.ndarray) -> "AffineScaler":
        values = _immutable_array(training_values, ndim=2, name="training_values")
        if values.shape[0] == 0:
            raise ValueError("Cannot fit a scaler on an empty training set.")
        return cls(values.min(axis=0), values.max(axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values_array = np.asarray(values)
        if values_array.shape[-1] != self.minimum.size:
            raise ValueError("Input column count does not match this scaler.")
        return 2.0 * (values_array - self.minimum) / (self.maximum - self.minimum) - 1.0

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values_array = np.asarray(values)
        if values_array.shape[-1] != self.minimum.size:
            raise ValueError("Input column count does not match this scaler.")
        return 0.5 * (values_array + 1.0) * (self.maximum - self.minimum) + self.minimum

    def state_dict(self) -> dict[str, list[float]]:
        return {"minimum": self.minimum.tolist(), "maximum": self.maximum.tolist()}


class SampleDataset4D(Dataset[dict[str, torch.Tensor]]):
    """Conventional dataset: ``len`` is rows and one item is exactly one row."""

    def __init__(
        self,
        bundle: FieldBundle4D,
        coordinate_scaler: AffineScaler,
        target_scaler: AffineScaler,
    ) -> None:
        self.bundle = bundle
        self.coordinate_scaler = coordinate_scaler
        self.target_scaler = target_scaler
        self.coordinates = torch.as_tensor(
            coordinate_scaler.transform(bundle.coordinates), dtype=torch.float32
        )
        self.targets = torch.as_tensor(
            target_scaler.transform(bundle.targets), dtype=torch.float32
        )

    def __len__(self) -> int:
        return self.bundle.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return {
            "coords": self.coordinates[index],
            "values": self.targets[index],
            "index": torch.tensor(index, dtype=torch.int64),
        }
