"""Read-only PFISR 4D observation adapter with actual record integration metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .data import FieldBundle4D


def _kilometres(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError(f"{name} has no finite values.")
    return values / 1000.0 if np.median(np.abs(finite)) > 2000.0 else values


def _beam_range(values: np.ndarray, beams: int, ranges: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape == (beams, ranges):
        return values
    if values.shape == (ranges, beams):
        return values.T
    raise ValueError(f"{name} shape {values.shape} is incompatible with {(beams, ranges)}.")


@dataclass(frozen=True)
class PFISRMetadata:
    path: str
    records: int
    beams: int
    ranges: int
    unix_start: np.ndarray
    unix_end: np.ndarray
    unix_mid: np.ndarray
    integration_duration_sec: np.ndarray

    def state_dict(self) -> dict[str, Any]:
        values, counts = np.unique(self.integration_duration_sec, return_counts=True)
        return {
            "path": self.path,
            "records": self.records,
            "beams": self.beams,
            "ranges": self.ranges,
            "first_record_start_unix": float(self.unix_start[0]),
            "last_record_end_unix": float(self.unix_end[-1]),
            "integration_duration_sec": {
                "nominal_product_median": float(np.median(self.integration_duration_sec)),
                "minimum": float(self.integration_duration_sec.min()),
                "maximum": float(self.integration_duration_sec.max()),
                "unique": np.unique(self.integration_duration_sec).tolist(),
                "counts": {str(float(value)): int(count) for value, count in zip(values, counts)},
                "final_record_duration": float(self.integration_duration_sec[-1]),
            },
            "midpoints_strictly_increasing": bool(np.all(np.diff(self.unix_mid) > 0)),
        }


@dataclass(frozen=True)
class PFISRReadConfig:
    altitude_min_km: float = 100.0
    altitude_max_km: float = 500.0
    minimum_ne_m3: float = 1.0e8
    time_start_unix: float | None = None
    time_end_unix: float | None = None
    record_start_index: int | None = None
    record_count: int | None = None
    record_stride: int = 1
    max_relative_uncertainty: float | None = None
    minimum_integration_fraction_of_file_median: float | None = None

    def validate(self) -> None:
        if self.altitude_max_km <= self.altitude_min_km or self.minimum_ne_m3 <= 0:
            raise ValueError("PFISR altitude or density bounds are invalid.")
        if self.record_stride <= 0:
            raise ValueError("record_stride must be positive.")
        if (self.record_start_index is None) != (self.record_count is None):
            raise ValueError("record_start_index and record_count must be supplied together.")
        if self.record_start_index is not None and (self.record_start_index < 0 or self.record_count <= 0):
            raise ValueError("Record index window is invalid.")
        if self.time_start_unix is not None and self.time_end_unix is not None:
            if self.time_end_unix < self.time_start_unix:
                raise ValueError("PFISR time bounds are reversed.")
        if self.max_relative_uncertainty is not None and self.max_relative_uncertainty <= 0:
            raise ValueError("max_relative_uncertainty must be positive.")
        if self.minimum_integration_fraction_of_file_median is not None:
            if not 0 < self.minimum_integration_fraction_of_file_median <= 1:
                raise ValueError("minimum integration fraction must be in (0, 1].")


@dataclass(frozen=True)
class PFISRCase4D:
    bundle: FieldBundle4D
    uncertainty_ne_m3: np.ndarray
    relative_uncertainty: np.ndarray
    uncertainty_log10: np.ndarray
    unix_start: np.ndarray
    unix_end: np.ndarray
    unix_mid: np.ndarray
    integration_duration_sec: np.ndarray
    range_km: np.ndarray
    range_indices: np.ndarray
    record_indices: np.ndarray
    exclusions: dict[str, int]
    metadata: PFISRMetadata


def inspect_pfisr_hdf5(path: Path) -> PFISRMetadata:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        required = (
            "BeamCodes",
            "FittedParams/Ne",
            "FittedParams/Range",
            "FittedParams/Altitude",
            "Time/UnixTime",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"Missing required PFISR datasets: {missing}")
        beam_codes = handle["BeamCodes"]
        ne = handle["FittedParams/Ne"]
        unix = np.asarray(handle["Time/UnixTime"], dtype=float)
        if beam_codes.ndim != 2 or beam_codes.shape[1] < 3:
            raise ValueError("BeamCodes must have at least beamcode, azimuth, and elevation.")
        if ne.ndim != 3 or unix.shape != (ne.shape[0], 2):
            raise ValueError("PFISR Ne/UnixTime shapes are inconsistent.")
        if not np.all(np.isfinite(unix)) or np.any(unix[:, 1] <= unix[:, 0]):
            raise ValueError("PFISR record start/end times are invalid.")
        midpoint = unix.mean(axis=1)
        if not np.all(np.diff(midpoint) > 0):
            raise ValueError("PFISR record midpoint times are not strictly increasing.")
        return PFISRMetadata(
            path=str(path.resolve()),
            records=ne.shape[0],
            beams=ne.shape[1],
            ranges=ne.shape[2],
            unix_start=unix[:, 0],
            unix_end=unix[:, 1],
            unix_mid=midpoint,
            integration_duration_sec=unix[:, 1] - unix[:, 0],
        )


def common_physical_window(
    left: PFISRMetadata,
    right: PFISRMetadata,
    *,
    requested_start_unix: float | None = None,
    requested_end_unix: float | None = None,
) -> tuple[float, float]:
    start = max(float(left.unix_start[0]), float(right.unix_start[0]))
    end = min(float(left.unix_end[-1]), float(right.unix_end[-1]))
    if requested_start_unix is not None:
        start = max(start, float(requested_start_unix))
    if requested_end_unix is not None:
        end = min(end, float(requested_end_unix))
    if end < start:
        raise ValueError("The PFISR products have no common requested physical time window.")
    return start, end


def _selected_records(metadata: PFISRMetadata, config: PFISRReadConfig) -> np.ndarray:
    if config.record_start_index is not None:
        end = config.record_start_index + int(config.record_count)
        if end > metadata.records:
            raise ValueError("Requested PFISR record index window exceeds the file.")
        records = np.arange(config.record_start_index, end)
    else:
        mask = np.ones(metadata.records, dtype=bool)
        if config.time_start_unix is not None:
            mask &= metadata.unix_mid >= config.time_start_unix
        if config.time_end_unix is not None:
            mask &= metadata.unix_mid <= config.time_end_unix
        records = np.flatnonzero(mask)
    records = records[:: config.record_stride]
    if len(records) == 0:
        raise ValueError("No PFISR records satisfy the selection.")
    return records


def read_pfisr_4d(path: Path, config: PFISRReadConfig) -> PFISRCase4D:
    """Read selected records without fitting preprocessing or replacing uncertainty."""

    config.validate()
    metadata = inspect_pfisr_hdf5(path)
    records = _selected_records(metadata, config)
    with h5py.File(path, "r") as handle:
        ne = np.asarray(handle["FittedParams/Ne"][records], dtype=float)
        dne = (
            np.asarray(handle["FittedParams/dNe"][records], dtype=float)
            if "FittedParams/dNe" in handle
            else np.full(ne.shape, np.nan)
        )
        beam_codes = np.asarray(handle["BeamCodes"], dtype=float)
        ranges = _beam_range(
            _kilometres(np.asarray(handle["FittedParams/Range"]), "Range"),
            metadata.beams,
            metadata.ranges,
            "Range",
        )
        altitudes = _beam_range(
            _kilometres(np.asarray(handle["FittedParams/Altitude"]), "Altitude"),
            metadata.beams,
            metadata.ranges,
            "Altitude",
        )

    azimuth = np.deg2rad(beam_codes[:, 1])[:, None]
    elevation = np.deg2rad(beam_codes[:, 2])[:, None]
    x = ranges * np.cos(elevation) * np.sin(azimuth)
    y = ranges * np.cos(elevation) * np.cos(azimuth)
    flat_ne = ne.reshape(-1)
    flat_dne = dne.reshape(-1)
    beam_index = np.broadcast_to(
        np.arange(metadata.beams)[None, :, None], ne.shape
    ).reshape(-1)
    range_index = np.broadcast_to(
        np.arange(metadata.ranges)[None, None, :], ne.shape
    ).reshape(-1)
    selected_record = np.broadcast_to(records[:, None, None], ne.shape).reshape(-1)
    flat_x = np.broadcast_to(x[None, :, :], ne.shape).reshape(-1)
    flat_y = np.broadcast_to(y[None, :, :], ne.shape).reshape(-1)
    flat_z = np.broadcast_to(altitudes[None, :, :], ne.shape).reshape(-1)
    flat_range = np.broadcast_to(ranges[None, :, :], ne.shape).reshape(-1)

    masks: dict[str, np.ndarray] = {}
    masks["nonfinite_density"] = ~np.isfinite(flat_ne)
    masks["density_below_minimum"] = np.isfinite(flat_ne) & (flat_ne <= config.minimum_ne_m3)
    masks["outside_altitude"] = ~(
        np.isfinite(flat_z)
        & (flat_z >= config.altitude_min_km)
        & (flat_z <= config.altitude_max_km)
    )
    coordinate_invalid = ~(
        np.isfinite(flat_x) & np.isfinite(flat_y) & np.isfinite(flat_z)
    )
    masks["nonfinite_coordinate"] = coordinate_invalid
    if config.minimum_integration_fraction_of_file_median is not None:
        duration_by_sample = metadata.integration_duration_sec[selected_record]
        minimum_duration = (
            config.minimum_integration_fraction_of_file_median
            * float(np.median(metadata.integration_duration_sec))
        )
        masks["incomplete_integration_record"] = duration_by_sample < minimum_duration
    relative = flat_dne / flat_ne
    if config.max_relative_uncertainty is not None:
        masks["invalid_uncertainty_quality"] = ~(
            np.isfinite(flat_dne)
            & (flat_dne > 0)
            & np.isfinite(relative)
            & (relative > 0)
            & (relative <= config.max_relative_uncertainty)
        )
    rejected = np.zeros(len(flat_ne), dtype=bool)
    exclusions: dict[str, int] = {}
    for name, mask in masks.items():
        newly_rejected = mask & ~rejected
        exclusions[name] = int(newly_rejected.sum())
        rejected |= mask
    valid = ~rejected
    if not np.any(valid):
        raise ValueError("PFISR filters rejected all selected observations.")

    selected_record_valid = selected_record[valid]
    unix_start = metadata.unix_start[selected_record_valid]
    unix_end = metadata.unix_end[selected_record_valid]
    unix_mid = metadata.unix_mid[selected_record_valid]
    integration_duration = metadata.integration_duration_sec[selected_record_valid]
    first_midpoint = float(metadata.unix_mid[records[0]])
    beam_ids = beam_codes[beam_index[valid], 0]
    time_ids = selected_record_valid
    relative_valid = relative[valid]
    uncertainty_log = relative_valid / np.log(10.0)
    coordinates = np.column_stack(
        [flat_x[valid], flat_y[valid], flat_z[valid], unix_mid - first_midpoint]
    )
    bundle = FieldBundle4D(
        coordinates=coordinates,
        targets=np.log10(flat_ne[valid])[:, None],
        beam_ids=beam_ids,
        time_ids=time_ids,
        group_ids=np.array([f"{beam}:{record}" for beam, record in zip(beam_ids, time_ids)]),
        metadata={
            "source": str(Path(path).resolve()),
            "selected_record_indices": records.tolist(),
            "coordinate_order": ["x_km", "y_km", "altitude_km", "t_sec_from_first_selected_midpoint"],
            "integration_duration_source": "Time/UnixTime record end minus start",
            "uncertainty_policy": "raw values retained; no replacement or weighting applied",
        },
    )
    exclusions["retained"] = int(valid.sum())
    return PFISRCase4D(
        bundle=bundle,
        uncertainty_ne_m3=flat_dne[valid],
        relative_uncertainty=relative_valid,
        uncertainty_log10=uncertainty_log,
        unix_start=unix_start,
        unix_end=unix_end,
        unix_mid=unix_mid,
        integration_duration_sec=integration_duration,
        range_km=flat_range[valid],
        range_indices=range_index[valid],
        record_indices=selected_record_valid,
        exclusions=exclusions,
        metadata=metadata,
    )
