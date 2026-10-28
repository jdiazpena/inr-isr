"""Native HDF observations, explicit UTC selection and strict real-data contracts.

The CSV is an audit copy. Training still constructs RadarTimeH5Dataset and uses
its unchanged float64 normalization / float32 tensors. No quality threshold or
uniform synthetic cadence is inferred here.
"""
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .window import ExperimentWindow


def unix(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def native_dataset(path, training):
    from datasets import RadarTimeH5Dataset
    return RadarTimeH5Dataset(path, h0_km=training["h0_km"],
        half_width_km=training["half_width_km"],
        time_start_utc=training["time_start_utc"], time_end_utc=training["time_end_utc"],
        window_start_index=training["window_start_index"],
        window_size_records=training["window_size_records"],
        record_stride=training["record_stride"], max_records=None, verbose=False)


def source_summary(path, training, dataset):
    """Describe native gates, exposures and every omitted time/beam gate pair."""
    from amisr_h5_reader_3d import _select_time_indices, _select_range_gate_per_beam, _to_km
    df = dataset.df
    with h5py.File(path, "r") as source:
        times = source["Time/UnixTime"][()].astype(float)
        selected = _select_time_indices(times, time_start_utc=training["time_start_utc"],
            time_end_utc=training["time_end_utc"], record_stride=training["record_stride"],
            max_records=None, window_start_index=training["window_start_index"],
            window_size_records=training["window_size_records"])
        altitude = _to_km(source["FittedParams/Altitude"][()], name="FittedParams/Altitude")
        ne_shape = source["FittedParams/Ne"].shape
        from amisr_h5_reader_3d import _force_beam_range_shape
        altitude = _force_beam_range_shape(altitude, ne_shape[1], ne_shape[2], "Altitude")
        beams, gates = _select_range_gate_per_beam(altitude, training["h0_km"], training["half_width_km"])
        present = set(map(tuple, df[["time_index", "beam_index", "range_index"]].to_numpy(dtype=int)))
        removed = []
        for time in selected:
            for beam, gate in zip(beams, gates):
                if (int(time), int(beam), int(gate)) not in present:
                    value = float(source["FittedParams/Ne"][time, beam, gate])
                    reason = "nonfinite_Ne" if not np.isfinite(value) else "nonpositive_Ne" if value <= 0 else "nonfinite_coordinate_or_log_target"
                    removed.append({"time_index":int(time), "beam_index":int(beam),
                                    "range_index":int(gate), "reason":reason})
        mid = times.mean(axis=1)
        origin = float(mid[selected[0]])
        return {"available_beams":int(len(source["BeamCodes"])),
            "selected_gate_beams":beams.tolist(), "selected_gate_indices":gates.tolist(),
            "source_record_indices":selected.tolist(), "source_records":len(selected),
            "retained_records":int(df.time_index.nunique()), "retained_beams":int(df.beam_index.nunique()),
            "possible_gate_rows":int(len(selected)*len(beams)), "retained_rows":len(df),
            "excluded_rows":removed, "unix_origin_sec":origin,
            "observed_midpoint_unix_bounds":[float(df.unix_mid.min()),float(df.unix_mid.max())],
            "exposure_unix_bounds":[float(times[selected,0].min()),float(times[selected,1].max())],
            "exposure_duration_sec":(times[selected,1]-times[selected,0]).tolist(),
            "midpoint_gaps_sec":np.diff(mid[selected]).tolist(),
            "selected_altitude_km_bounds":[float(altitude[beams,gates].min()),float(altitude[beams,gates].max())]}


def validate_radar_inputs(contract, dataset=None):
    """Reject missing rows, changed native selection, invented times or silent filters."""
    data = contract.data
    training = data["training"]
    if not contract.radar or data.get("selection_every") != 0:
        raise ValueError("Real reconstruction requires final-update saving without score selection")
    if any(name in data["inputs"] for name in ("development", "withheld", "reserved", "scene")):
        raise ValueError("Real all-beam reconstruction has no held tables or synthetic truth")
    if training["window_start_index"] is not None or training["window_size_records"] is not None:
        raise ValueError("UTC selection requires both record-index selectors to be None")
    start, end = map(unix, (training["time_start_utc"], training["time_end_utc"]))
    maximum = data["window_policy"]["maximum_duration_sec"]
    if not np.isfinite([start,end,maximum]).all() or not 0 < end-start <= maximum or maximum > 3600:
        raise ValueError("Radar clock selection must respect the one-hour maximum")
    if training["record_stride"] != 1 or data.get("measurement_policy") != "native_exposure_average_at_midpoint":
        raise ValueError("This all-observation route requires stride one and native averaged targets")
    source = contract.file("hdf5")
    if Path(training["h5_path"]).resolve() != source:
        raise ValueError("Native HDF path differs from the contracted source")
    if dataset is None:
        dataset = native_dataset(source, training)
    if Path(dataset.h5_path).resolve() != source:
        raise ValueError("Dataset belongs to another radar source")
    actual = source_summary(source, training, dataset)
    if actual != data["observation_summary"]:
        raise ValueError("Native selection, exclusions or timestamps differ from their declaration")
    copied = pd.read_csv(contract.file("train"), float_precision="round_trip")
    if list(copied.columns) != list(dataset.df.columns) or not np.array_equal(
            copied.to_numpy(), dataset.df.to_numpy(), equal_nan=True):
        raise ValueError("Audit table must contain every retained native observation exactly once")
    geometry = pd.read_csv(contract.file("roles"), float_precision="round_trip")
    expected = dataset.df[["beam_index","beamcode","az_deg","el_deg","range_index","altitude_km","x_km","y_km"]].drop_duplicates().sort_values("beam_index")
    if (geometry.beam_index.duplicated().any() or not geometry.role.eq("train").all()
            or set(data["roles"]["train"]) != set(expected.beam_index)
            or actual["available_beams"] != len(expected)
            or not np.array_equal(geometry[list(expected)].to_numpy(), expected.to_numpy(), equal_nan=True)):
        raise ValueError("All available native beam locations must be declared and used")
    window = ExperimentWindow.from_contract(contract).validate_inputs(contract)
    window.assert_scalers(dataset.coord_scalers, True)
    if data["preprocessing"] != {"coord_scalers":dataset.coord_scalers,"target_scaler":dataset.target_scaler}:
        raise ValueError("Contract normalization differs from native selected-window normalization")
    if data["time_bounds_sec"] != [float(dataset.df.t_sec.min()),float(dataset.df.t_sec.max())]:
        raise ValueError("Model time bounds must equal actual retained midpoint support")
    if (dataset.df.unix_mid.min() < start or dataset.df.unix_mid.max() > end
            or not np.array_equal(dataset.df.t_sec.to_numpy() + actual["unix_origin_sec"], dataset.df.unix_mid.to_numpy())):
        raise ValueError("Native local time / UTC origin mismatch")
    for name, expected_count in (("probes",training["num_diagnostic_collocation"]),
                                 ("queries",data["analysis"]["field_query_count"])):
        frame = pd.read_csv(contract.file(name), float_precision="round_trip")
        if len(frame) != expected_count or not np.isfinite(frame[["x_km","y_km","t_sec"]].to_numpy()).all():
            raise ValueError(f"Invalid {name} count or coordinates")
        if not np.allclose(frame.unix_mid, frame.t_sec + actual["unix_origin_sec"], rtol=0,atol=1e-7):
            raise ValueError(f"Invalid {name} UTC coordinates")
    return dataset
