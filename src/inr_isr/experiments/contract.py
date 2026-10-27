"""Checksummed input contracts and group-balanced development scores."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Contract:
    path: Path
    data: dict

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if data.get("schema") not in ("shared-xyt-development-1", "shared-xyt-reconstruction-1", "shared-xyt-withholding-1"):
            raise ValueError("Unknown experiment contract")
        roles = [set(data["roles"][name]) for name in ("train", "development", "reserved")]
        if data["schema"] == "shared-xyt-development-1":
            if any(not role for role in roles) or any(roles[i] & roles[j] for i in range(3) for j in range(i)):
                raise ValueError("Roles must be nonempty and disjoint")
        elif not roles[0] or roles[1] or roles[2] or data.get("checkpoint_policy") != "final_update":
            raise ValueError("All-beam reconstruction requires all training, zero withheld and final-update saving")
        contract = cls(path, data)
        for name in data["inputs"]:
            contract.file(name)
        if contract.all_beam:
            contract.validate_all_beam_inputs()
        elif contract.withholding:
            contract.validate_withholding_inputs()
        return contract

    @property
    def all_beam(self):
        return self.data.get("schema") == "shared-xyt-reconstruction-1"

    @property
    def withholding(self):
        return self.data.get("schema") == "shared-xyt-withholding-1"

    @property
    def final_update(self):
        return self.all_beam or self.withholding

    def validate_withholding_inputs(self):
        """An exact, explicitly declared observation partition; never a selection set."""
        import pandas as pd
        if not self.withholding or self.data.get("selection_every") != 0:
            raise ValueError("Withholding assessment requires final-update saving without score selection")
        geometry = pd.read_csv(self.file("roles"))
        source = pd.read_csv(self.file("source_observations"))
        train = pd.read_csv(self.file("train"))
        held = pd.read_csv(self.file("withheld"))
        if "development" in self.data["inputs"] or "reserved" in self.data["inputs"]:
            raise ValueError("Assessment-only inputs must not introduce development/reserved groups")
        available = set(geometry.beam_index)
        if geometry.beam_index.duplicated().any() or not available:
            raise ValueError("Geometry locations must be unique and nonempty")
        start, end = self.data["time_bounds_sec"]
        cadence = self.data["observation_cadence_sec"]
        intervals = (end - start) / cadence if np.isfinite(cadence) and cadence > 0 else np.nan
        if not np.isfinite(intervals) or intervals < 1 or not np.isclose(intervals, round(intervals), rtol=0, atol=1e-10):
            raise ValueError("Source cadence and window endpoints must agree")
        times = start + np.arange(round(intervals) + 1) * cadence
        core = ["x_km", "y_km", "t_sec", "log10_Ne"]
        keys = ["beam_id", "t_sec"]
        for name, frame in (("source", source), ("train", train), ("withheld", held)):
            if not len(frame) or frame.duplicated(keys).any() or not np.isfinite(frame[core].to_numpy()).all():
                raise ValueError(f"Invalid or duplicated {name} observations")
        if (set(source.beam_id) != available or len(source) != len(available) * len(times)
                or not np.array_equal(np.sort(source.t_sec.unique()), times)
                or not source.groupby("beam_id").size().eq(len(times)).all()):
            raise ValueError("Source observations must cover the complete declared geometry and cadence")
        expected_xy = geometry.set_index("beam_index")[["x_km", "y_km"]].loc[source.beam_id].to_numpy()
        if not np.allclose(source[["x_km", "y_km"]].to_numpy(), expected_xy, rtol=0, atol=1e-10):
            raise ValueError("Source positions differ from declared geometry")
        rule = self.data["withholding"]
        beams, excluded_times = rule["beam_indices"], rule["times_sec"]
        if len(set(beams)) != len(beams) or len(set(excluded_times)) != len(excluded_times):
            raise ValueError("Duplicate exclusion declarations")
        if rule["kind"] == "beams":
            if not beams or excluded_times or not set(beams) < available:
                raise ValueError("Beam-only assessment requires a proper beam subset and no removed times")
            mask = source.beam_id.isin(beams)
        elif rule["kind"] == "times":
            if beams or not excluded_times or not set(excluded_times).issubset(set(times)) or any(t <= start or t >= end for t in excluded_times):
                raise ValueError("Time-only assessment requires declared interior times and no removed beams")
            mask = source.t_sec.isin(excluded_times)
        else:
            raise ValueError("Declare a separate beam-only or time-only assessment")
        for actual, expected in ((train, source.loc[~mask]), (held, source.loc[mask])):
            a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
            if not a.index.equals(b.index) or not np.array_equal(a[core[:-2] + ["log10_Ne"]].to_numpy(), b[core[:-2] + ["log10_Ne"]].to_numpy()):
                raise ValueError("Training/withheld tables must exactly partition the declared source observations")
        training_beams = available - set(beams)
        expected_roles = np.where(geometry.beam_index.isin(beams), "withheld", "train")
        if (set(self.data["roles"]["train"]) != training_beams
                or set(train.beam_id) != training_beams
                or not np.array_equal(geometry.role.to_numpy(), expected_roles)):
            raise ValueError("Training beam roles disagree with the observation partition")

    def validate_all_beam_inputs(self):
        """Reject omitted locations/times, including an omitted complete cadence record."""
        import pandas as pd
        if "development" in self.data["inputs"]:
            raise ValueError("All-beam reconstruction must not supply a development table")
        geometry = pd.read_csv(self.file("roles"))
        train = pd.read_csv(self.file("train"))
        available = set(geometry.beam_index)
        if (geometry.beam_index.duplicated().any() or not geometry.role.eq("train").all()
                or set(self.data["roles"]["train"]) != available or set(train.beam_id) != available):
            raise ValueError("All available geometry locations must be used for training")
        start, end = self.data["time_bounds_sec"]
        cadence = self.data["observation_cadence_sec"]
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("Observation cadence must be positive and finite")
        n = (end - start) / cadence
        if n < 1 or not np.isclose(n, round(n), rtol=0, atol=1e-10):
            raise ValueError("Window endpoints must lie on the declared cadence")
        times = start + np.arange(round(n) + 1) * cadence
        if (train.duplicated(["beam_id", "t_sec"]).any()
                or len(train) != len(available) * len(times)
                or not np.array_equal(np.sort(train.t_sec.unique()), times)
                or not train.groupby("beam_id").size().eq(len(times)).all()):
            raise ValueError("All-beam input must contain every location at every declared observation time")
        if not np.isfinite(train[["x_km", "y_km", "t_sec", "log10_Ne"]].to_numpy()).all():
            raise ValueError("All-beam input contains nonfinite measurements")
        expected = geometry.set_index("beam_index")[["x_km", "y_km"]].loc[train.beam_id].to_numpy()
        if not np.allclose(train[["x_km", "y_km"]].to_numpy(), expected, rtol=0, atol=1e-10):
            raise ValueError("Measurement positions differ from the complete geometry")

    def file(self, name):
        record = self.data["inputs"][name]
        path = (self.path.parent / record["path"]).resolve()
        if digest(path) != record["sha256"]:
            raise ValueError(f"Changed experiment input: {name}: {path}")
        return path


def grouped_metrics(prediction, target, groups):
    prediction, target = np.asarray(prediction), np.asarray(target)
    error = prediction - target
    groups = np.asarray(groups)
    if error.ndim != 1 or groups.shape != error.shape or not len(error) or not np.isfinite(error).all():
        raise ValueError("Scores require finite nonempty one-dimensional paired values/groups")
    rows = []
    for group in np.unique(groups):
        e = error[groups == group]
        rows.append({"beam_id": int(group), "count": len(e), "mse": float(np.mean(e**2)),
                     "mae": float(np.mean(abs(e))), "bias": float(np.mean(e))})
    return {"macro_rmse_dex": float(np.sqrt(np.mean([r["mse"] for r in rows]))),
            "pooled_rmse_dex": float(np.sqrt(np.mean(error**2))),
            "mae_dex": float(np.mean(abs(error))), "bias_dex": float(np.mean(error)),
            "per_beam": rows}


def improves(score, best):
    if not np.isfinite(score):
        raise ValueError("Nonfinite development score")
    return score < best  # An exact tie keeps the earlier checkpoint.
