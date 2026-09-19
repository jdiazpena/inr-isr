"""Named physical observations: representation is not a training or loss policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .arrays import immutable_array

AXES = {"xyt": ("x_km", "y_km", "t_sec"),
        "xyzt": ("x_km", "y_km", "z_km", "t_sec")}
UNITS = {"xyt": ("km", "km", "s"), "xyzt": ("km", "km", "km", "s")}


@dataclass(frozen=True)
class PhysicalField:
    """Physical coordinates[N,D], log10(Ne/m^-3) targets[N,1] and source identities.

Coordinates/targets/identity arrays are owned read-only copies. metadata is a
shallow dictionary copy and may retain a native mutable dataframe/config object;
it is NOT deeply frozen. Native dataset/problem objects remain separately usable.
No radar truth or groups are fabricated. Identifiers are copied as supplied, not
normalized, sorted or coerced. Legacy nonfinite retention must be explicit.
For PFISR xyzt, z_km denotes product altitude, not a silently substituted range.
"""
    source: str
    coordinate_mode: str
    coordinates: np.ndarray
    targets: np.ndarray
    row_identity: Mapping[str, np.ndarray]
    metadata: Mapping[str, object] = field(default_factory=dict)
    allow_nonfinite: bool = False

    def __post_init__(self) -> None:
        """Validate named axes/single-target alignment without altering native data."""
        if self.source not in {"radar", "synthetic"} or self.coordinate_mode not in AXES:
            raise ValueError("PhysicalField requires source radar/synthetic and mode xyt/xyzt.")
        coordinates = immutable_array(self.coordinates, ndim=2, name="coordinates",
                                      require_finite=not self.allow_nonfinite)
        targets = immutable_array(self.targets, ndim=2, name="targets",
                                  require_finite=not self.allow_nonfinite)
        if coordinates.shape[1] != len(AXES[self.coordinate_mode]):
            raise ValueError("Coordinate columns do not match the named mode.")
        if targets.shape != (len(coordinates), 1) or not len(coordinates):
            raise ValueError("PhysicalField requires nonempty aligned [N,1] targets.")
        identifiers = {}
        for name, values in self.row_identity.items():
            array = np.asarray(values).copy()
            if array.ndim != 1 or len(array) != len(coordinates):
                raise ValueError(f"Row identity {name} must have shape [N].")
            array.setflags(write=False)
            identifiers[name] = array
        if not identifiers:
            raise ValueError("At least one documented source row identity is required.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "row_identity", identifiers)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def axes(self) -> tuple[str, ...]:
        """Return physical column names in the actual model-input order."""
        return AXES[self.coordinate_mode]

    @property
    def units(self) -> tuple[str, ...]:
        """Return each coordinate's physical unit, before affine normalization."""
        return UNITS[self.coordinate_mode]

    @property
    def target_transform(self) -> str:
        """Name the physical target, distinct from the later affine training map."""
        return "log10(Ne / (1 m^-3))"
