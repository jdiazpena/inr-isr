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
        if data.get("schema") != "shared-xyt-development-1":
            raise ValueError("Unknown experiment contract")
        roles = [set(data["roles"][name]) for name in ("train", "development", "reserved")]
        if any(not role for role in roles) or any(roles[i] & roles[j] for i in range(3) for j in range(i)):
            raise ValueError("Roles must be nonempty and disjoint")
        contract = cls(path, data)
        for name in data["inputs"]:
            contract.file(name)
        return contract

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
