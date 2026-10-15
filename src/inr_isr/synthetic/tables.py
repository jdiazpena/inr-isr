"""Compatible sparse tables from a frozen scene; existing dataset remains the reader."""
import numpy as np
import pandas as pd

from synthetic_plasma import make_time_array
from .field import scene_from_definition, average_scene


def observation_table(definition, geometry, speed_km_s, *, start_sec=0, end_sec=None,
                      exposure_sec=0, integration_spacing_sec=0.1):
    """Keep original record indices when selecting a shorter elapsed-time window."""
    duration = definition["duration_sec"]
    cadence = definition["observation_cadence_sec"]
    end_sec = duration if end_sec is None else end_sec
    if not (0 <= start_sec <= end_sec <= duration) or cadence <= 0:
        raise ValueError("Invalid observation window/cadence")
    records = duration / cadence
    if not np.isclose(records, round(records), rtol=0, atol=1e-10):
        raise ValueError("Duration must be an integer number of cadence intervals")
    times = make_time_array(duration, int(round(records)) + 1)
    ids = np.flatnonzero((times >= start_sec) & (times <= end_sec))
    if not ids.size:
        raise ValueError("No original observation records in requested window")
    n = len(geometry)
    if not n or not np.isfinite(geometry[["x_km", "y_km"]].to_numpy()).all():
        raise ValueError("Geometry must contain finite locations")
    df = pd.concat([geometry] * len(ids), ignore_index=True)
    df["beam_id"] = np.tile(geometry["beam_index"].to_numpy(), len(ids))
    df["time_index"] = np.repeat(ids, n)
    df["t_sec"] = np.repeat(times[ids], n)
    scene = scene_from_definition(definition, speed_km_s)
    vals = average_scene(scene, df.x_km.to_numpy(), df.y_km.to_numpy(),
                         df.t_sec.to_numpy(), exposure_sec,
                         max_spacing_sec=integration_spacing_sec)
    for key in ("Ne", "log10_Ne", "true_dlog10Ne_dx_km", "true_dlog10Ne_dy_km",
                "true_dlog10Ne_dt_sec"):
        df[key] = vals[key]
    df["log10_Ne_true"] = df.log10_Ne
    df["integration_time_sec"] = exposure_sec
    df["integration_start_sec"] = df.t_sec - exposure_sec / 2
    df["integration_end_sec"] = df.t_sec + exposure_sec / 2
    df["sample_geometry"] = "saved_radar_footprint"
    df["is_observed"] = True
    return df
