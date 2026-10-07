"""Strict executable case schema distinct from historical proposal documents."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path
import json

from .common import canonical_hash, sha256

TOP = {"schema_version", "case_id", "input", "tables", "field", "bounds", "queries", "training", "diagnostics"}
TRAINING = {"model", "seed", "lr", "microbatch_observations", "max_updates", "max_seconds", "validation_every", "device", "dtype", "amp", "regularization"}


def keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) != set(allowed):
        raise ValueError(f"{name} must contain exactly {sorted(allowed)}")


def positive(value, name, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


def validate(raw):
    keys(raw, TOP, "case")
    if raw["schema_version"] != 1 or not isinstance(raw["case_id"], str) or not raw["case_id"]:
        raise ValueError("Invalid schema_version/case_id")
    keys(raw["input"], {"path", "sha256", "reader_sha256", "h0_km", "half_width_km", "utc_start", "utc_end"}, "input")
    keys(raw["tables"], {"geometry", "times"}, "tables")
    for name, table in raw["tables"].items():
        keys(table, {"path", "sha256"}, f"table {name}")
    for item in [raw["input"], *raw["tables"].values(), *raw["queries"].values()]:
        for key, value in item.items():
            if key.endswith("sha256") and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError(f"Invalid identity: {key}")
    keys(raw["field"], {"background_ne_m3", "patch", "noise_std_log10"}, "field")
    keys(raw["field"]["patch"], {"name", "amplitude_m3", "sigma_x_km", "sigma_y_km", "x0_km", "y0_km", "vx_km_s", "vy_km_s"}, "patch")
    positive(raw["field"]["background_ne_m3"], "background")
    for name in ("amplitude_m3", "sigma_x_km", "sigma_y_km"):
        positive(raw["field"]["patch"][name], name)
    for name in ("x0_km", "y0_km", "vx_km_s", "vy_km_s"):
        if type(raw["field"]["patch"][name]) not in (int, float) or not math.isfinite(raw["field"]["patch"][name]):
            raise ValueError(f"Nonfinite field parameter: {name}")
    if raw["field"]["noise_std_log10"] != 0:
        raise ValueError("This first results route implements the clean reference only")
    if len(raw["bounds"]) != 3 or any(len(b) != 2 or not all(math.isfinite(x) for x in b) or b[1] <= b[0] for b in raw["bounds"]):
        raise ValueError("bounds must contain three finite increasing pairs")
    keys(raw["queries"], {"development", "final"}, "queries")
    for role, q in raw["queries"].items():
        keys(q, {"count", "seed", "coordinate_sha256"}, f"queries {role}")
        positive(q["count"], "query count", True)
        if type(q["seed"]) is not int or q["seed"] < 0:
            raise ValueError("Query seed must be a nonnegative integer")
    keys(raw["training"], TRAINING, "training")
    training = raw["training"]
    keys(training["model"], {"activation", "hidden_features", "hidden_layers", "first_omega_0", "hidden_omega_0"}, "model")
    if training["model"]["activation"] != "sine":
        raise ValueError("This route requires the declared sine model")
    for name in ("hidden_features", "hidden_layers"):
        positive(training["model"][name], name, True)
    for name in ("first_omega_0", "hidden_omega_0"):
        positive(training["model"][name], name)
    for name in ("microbatch_observations", "max_updates", "validation_every"):
        positive(training[name], name, True)
    for name in ("lr", "max_seconds"):
        positive(training[name], name)
    if type(training["seed"]) is not int or training["seed"] < 0:
        raise ValueError("Training seed must be a nonnegative integer")
    if training["dtype"] != "float32" or training["amp"] is not False:
        raise ValueError("First pilot requires float32 with AMP disabled")
    if not isinstance(training["device"], str) or not re.fullmatch(r"cuda(?::[0-9]+)?", training["device"]):
        raise ValueError("First results pilot requires explicit CUDA")
    expected = {"lambda_curv_xy": 0.0, "lambda_curv_t": 0.0, "reference_loss_weights": False,
                "target_xy_ratio": 0.0, "target_t_ratio": 0.0, "reg_ramp_frac": 0.0}
    if training["regularization"] != expected:
        raise ValueError("Data-only route requires all fixed/adaptive priors disabled")
    keys(raw["diagnostics"], {"slice_fractions", "grid_size", "trace_beam_indices"}, "diagnostics")
    positive(raw["diagnostics"]["grid_size"], "grid_size", True)
    if not raw["diagnostics"]["slice_fractions"] or not all(0 < x < 1 for x in raw["diagnostics"]["slice_fractions"]):
        raise ValueError("Development slice fractions must be strictly internal")
    traces = raw["diagnostics"]["trace_beam_indices"]
    if not traces or any(type(i) is not int or i < 0 for i in traces) or len(set(traces)) != len(traces):
        raise ValueError("Diagnostic trace identities must be distinct nonnegative integers")
    return raw


def load_case(path):
    """Validate before data/model work, resolve paths relative to saved config."""
    path = Path(path).resolve()
    raw = validate(json.loads(path.read_text()))
    resolved = copy.deepcopy(raw)
    for item in [resolved["input"], *resolved["tables"].values()]:
        item["path"] = str((path.parent / item["path"]).resolve())
    return resolved, canonical_hash(raw), sha256(path)


def training_settings(case, smoke=False):
    settings = copy.deepcopy(case["training"])
    if smoke:
        settings.update(max_updates=2, max_seconds=120.0, validation_every=1)
    return settings
