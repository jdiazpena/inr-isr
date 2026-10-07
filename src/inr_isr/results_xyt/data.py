"""Metadata-driven clean xyt observations; reserved targets stay unopened."""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

import amisr_h5_reader_3d as reader
from synthetic_plasma import MovingGaussianPatch, evaluate_integration_averaged_plasma, evaluate_synthetic_plasma
from inr_isr.transforms import affine_forward, affine_inverse
from .common import canonical_hash, fresh_directory, sha256, write_json

NODES = 33


def read_tables(case, verify_input=True):
    tables = {}
    for name, item in case["tables"].items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Changed {name} table")
        tables[name] = pd.read_csv(item["path"], float_precision="round_trip")
    geometry, times = tables["geometry"], tables["times"]
    if not geometry.beam_index.is_unique or not geometry.beamcode.is_unique or not times.original_record_index.is_unique:
        raise ValueError("Duplicate source identities")
    if set(geometry.spatial_role) != {"train", "validation", "final_test"}:
        raise ValueError("Spatial roles must be nonempty and exactly train/validation/final_test")
    numeric = ["beam_index", "beamcode", "range_index", "x_km", "y_km"]
    if not np.isfinite(geometry[numeric]).all().all() or not np.isfinite(times[["unix_start", "unix_end", "unix_mid", "t_sec"]]).all().all():
        raise ValueError("Nonfinite metadata")
    if np.any(times.unix_end <= times.unix_start) or np.any(times.unix_start.to_numpy()[1:] < times.unix_end.to_numpy()[:-1]):
        raise ValueError("Invalid/overlapping integration intervals")
    if not np.array_equal(times.t_sec.to_numpy(), times.unix_mid.to_numpy() - times.unix_mid.iloc[0]):
        raise ValueError("Time origin mismatch")
    xy = geometry[["x_km", "y_km"]].to_numpy()
    if not np.all(Delaunay(xy[geometry.spatial_role == "train"]).find_simplex(xy) >= 0):
        raise ValueError("This spatial interpolation design requires all beams inside training hull")
    if verify_input:
        item = case["input"]
        if sha256(item["path"]) != item["sha256"] or sha256(reader.__file__) != item["reader_sha256"]:
            raise ValueError("Original input/reader identity changed")
        with h5py.File(item["path"], "r") as handle:
            _, nb, nr = handle["FittedParams/Ne"].shape  # Metadata only, no radar targets read.
            beams = np.asarray(handle["BeamCodes"], dtype=float)
            altitude = reader._force_beam_range_shape(reader._to_km(handle["FittedParams/Altitude"][()], "Altitude"), nb, nr, "Altitude")
            ranges = reader._force_beam_range_shape(reader._to_km(handle["FittedParams/Range"][()], "Range"), nb, nr, "Range")
            bi, ri = reader._select_range_gate_per_beam(altitude, item["h0_km"], item["half_width_km"])
            x, y, _ = reader._compute_radar_xyz_from_range(ranges, beams[:, 1], beams[:, 2])
            intervals = np.asarray(handle["Time/UnixTime"], dtype=float)
        mid = intervals.mean(axis=1)
        ids = np.flatnonzero((mid >= reader._parse_utc_to_unix(item["utc_start"])) & (mid <= reader._parse_utc_to_unix(item["utc_end"])))
        expected = {"beam_index": bi, "beamcode": beams[bi, 0].astype(int), "range_index": ri,
                    "x_km": x[bi, ri], "y_km": y[bi, ri]}
        for name, values in expected.items():
            if not np.array_equal(geometry[name].to_numpy(), values):
                raise ValueError(f"Geometry table differs from original metadata: {name}")
        for name, values in {"original_record_index": ids, "unix_start": intervals[ids, 0],
                             "unix_end": intervals[ids, 1], "unix_mid": mid[ids]}.items():
            if not np.array_equal(times[name].to_numpy(), values):
                raise ValueError(f"Time table differs from original metadata: {name}")
    return geometry, times


def metadata_rows(geometry, times):
    """Full metadata support is known; this function creates no density labels."""
    frame = times.merge(geometry, how="cross")
    origin = float(times.unix_mid.iloc[0])
    frame["start_sec"] = frame.unix_start - origin
    frame["end_sec"] = frame.unix_end - origin
    frame["role"] = frame.spatial_role
    frame["observation_id"] = [f"pfisr20120122:record={r.original_record_index}:beam={r.beamcode}:gate={r.range_index}" for r in frame.itertuples()]
    if not frame.observation_id.is_unique:
        raise ValueError("Nonunique observations")
    return frame


def field_values(case, coordinates):
    p = case["field"]
    return evaluate_synthetic_plasma(*np.asarray(coordinates).T, p["background_ne_m3"], [MovingGaussianPatch(**p["patch"])])["log10_Ne"]


def observation_truth(case, rows, allow_final=False):
    if (rows.role == "final_test").any() and not allow_final:
        raise ValueError("Reserved final targets require explicit frozen final evaluation")
    p = case["field"]
    result = np.empty(len(rows), dtype=np.float64)
    duration = (rows.end_sec - rows.start_sec).to_numpy()
    for value in np.unique(duration):
        mask = duration == value
        group = rows.iloc[np.flatnonzero(mask)]
        result[mask] = evaluate_integration_averaged_plasma(
            group.x_km.to_numpy(), group.y_km.to_numpy(), group.t_sec.to_numpy(),
            float(value), NODES, p["background_ne_m3"], [MovingGaussianPatch(**p["patch"])])["log10_Ne"]
    return result


def fit_scaler(log_targets, roles):
    """Validate and fit training values only; held-out values never affect limits."""
    values = np.asarray(log_targets)[np.asarray(roles) == "train"]
    if not len(values) or not np.isfinite(values).all() or np.ptp(values) <= 0:
        raise ValueError("Training targets must be nonempty, finite and nondegenerate")
    return {"min": float(values.min()), "max": float(values.max()), "fit_count": len(values)}


def normalize_targets(values, scaler):
    return affine_forward(np.asarray(values), scaler["min"], scaler["max"])


def inverse_targets(values, scaler):
    return affine_inverse(np.asarray(values), scaler["min"], scaler["max"])


def normalize_points(points, bounds):
    bounds = np.asarray(bounds, dtype=np.float64)
    return affine_forward(np.asarray(points), bounds[:, 0], bounds[:, 1]).astype(np.float32)


def exposure_points(rows):
    start, end = rows.start_sec.to_numpy(), rows.end_sec.to_numpy()
    t = start[:, None] + np.linspace(0, 1, NODES) * (end - start)[:, None]
    x = np.broadcast_to(rows.x_km.to_numpy()[:, None], t.shape)
    y = np.broadcast_to(rows.y_km.to_numpy()[:, None], t.shape)
    return np.stack([x, y, t], axis=-1)


def quadrature_weights():
    weights = np.ones(NODES, dtype=np.float64)
    weights[[0, -1]] = 0.5
    return weights / weights.sum()


def make_queries(case, geometry, times, role="development", allow_final=False):
    if role not in {"development", "final"} or (role == "final" and not allow_final):
        raise ValueError("Final queries require explicit frozen final evaluation")
    q = case["queries"][role]
    hull = Delaunay(geometry.loc[geometry.spatial_role == "train", ["x_km", "y_km"]].to_numpy())
    bounds = np.asarray(case["bounds"])
    lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
    lo[2], hi[2] = times.t_sec.min(), times.t_sec.max()
    rng = np.random.Generator(np.random.PCG64(q["seed"]))
    accepted = []
    while sum(len(a) for a in accepted) < q["count"]:
        candidates = lo + rng.random((256, 3)) * (hi - lo)
        accepted.append(candidates[hull.find_simplex(candidates[:, :2]) >= 0])
    points = np.concatenate(accepted)[:q["count"]]
    import hashlib
    if hashlib.sha256(points.astype("<f8").tobytes()).hexdigest() != q["coordinate_sha256"]:
        raise ValueError("Query coordinates differ from approved recipe")
    return points


def prepare(case, config_identity, output):
    geometry, times = read_tables(case)
    metadata = metadata_rows(geometry, times)
    rows = metadata[metadata.role.isin(["train", "validation"])].copy()
    rows["log10_Ne"] = observation_truth(case, rows)
    scaler = fit_scaler(rows.log10_Ne, rows.role)
    rows["normalized_target"] = normalize_targets(rows.log10_Ne, scaler)
    points = exposure_points(rows)
    bounds = np.asarray(case["bounds"])
    if np.any(points < bounds[:, 0]) or np.any(points > bounds[:, 1]):
        raise ValueError("Integration nodes lie outside declared coordinate bounds")
    queries = make_queries(case, geometry, times)
    output = fresh_directory(output)
    rows.to_csv(output / "development_observations.csv", index=False)
    metadata[metadata.role == "final_test"].to_csv(output / "reserved_metadata.csv", index=False)
    np.save(output / "development_coordinates.npy", queries, allow_pickle=False)
    record = {"format": "xyt_results_data_v1", "config_identity": config_identity,
              "scaler": scaler, "bounds": case["bounds"], "quadrature_nodes": NODES,
              "role_counts": metadata.role.value_counts().to_dict(), "final_targets_generated": False,
              "input_sha256": case["input"]["sha256"], "table_sha256": {k: v["sha256"] for k, v in case["tables"].items()},
              "artifacts": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "data_manifest.json", record)
    return record


def load_prepared(path, config_identity):
    import json
    path = Path(path)
    record = json.loads((path / "data_manifest.json").read_text())
    if record["format"] != "xyt_results_data_v1" or record["config_identity"] != config_identity or record["final_targets_generated"]:
        raise ValueError("Prepared data identity/role policy mismatch")
    for name, expected in record["artifacts"].items():
        if sha256(path / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    rows = pd.read_csv(path / "development_observations.csv", float_precision="round_trip")
    if set(rows.role) != {"train", "validation"} or not rows.observation_id.is_unique:
        raise ValueError("Prepared development rows contain invalid roles/identities")
    if fit_scaler(rows.log10_Ne, rows.role) != record["scaler"]:
        raise ValueError("Prepared scaler is not fitted on training rows")
    return rows, record
