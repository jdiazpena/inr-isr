"""A configured physical time window shared by preparation and evaluation.

Absolute start/end times never belong in an analysis function. A caller may
also declare a maximum duration in window_policy; current research contracts
declare their one-hour limit there. Legacy metadata without time_bounds_sec
can derive its bounds from the supplied training table for compatibility.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentWindow:
    start: float
    end: float
    maximum_duration: float | None = None

    def __post_init__(self):
        if not np.isfinite([self.start, self.end]).all() or self.end <= self.start:
            raise ValueError("Window requires finite increasing physical times")
        if self.maximum_duration is not None:
            if not np.isfinite(self.maximum_duration) or self.maximum_duration <= 0:
                raise ValueError("Maximum window duration must be finite and positive")
            if self.duration > self.maximum_duration:
                raise ValueError("Time window exceeds its configured maximum duration")

    @property
    def duration(self):
        return self.end - self.start

    @classmethod
    def from_contract(cls, contract):
        bounds = contract.data.get("time_bounds_sec")
        if bounds is None:
            times = pd.read_csv(contract.file("train"), usecols=["t_sec"]).t_sec
            bounds = [float(times.min()), float(times.max())]
        if len(bounds) != 2:
            raise ValueError("time_bounds_sec must contain exactly two bounds")
        return cls(*map(float, bounds), contract.data.get("window_policy", {}).get("maximum_duration_sec"))

    def assert_times(self, values, label="query"):
        times = np.asarray(values, dtype=float)
        if not np.isfinite(times).all() or ((times < self.start) | (times > self.end)).any():
            raise ValueError(f"{label} contains times outside the configured window")

    def sample_times(self, step, offset=0.):
        if not np.isfinite([step, offset]).all() or step <= 0 or not 0 <= offset < self.duration:
            raise ValueError("Sampling step/offset must fit inside the configured window")
        times = np.arange(self.start + offset, self.end, step)
        self.assert_times(times)
        return times

    def assert_scalers(self, scalers, require_endpoints=False):
        time = scalers['t_sec']
        self.assert_times([time['min'], time['max']], 'time normalization')
        if time['max'] <= time['min']:
            raise ValueError('Time normalization has no positive span')
        if require_endpoints and [time['min'], time['max']] != [self.start, self.end]:
            raise ValueError('Time normalization differs from the configured window endpoints')

    def validate_inputs(self, contract):
        for name in ("train", "development", "withheld", "source_observations", "probes", "queries"):
            if name not in contract.data["inputs"]:
                continue
            frame = pd.read_csv(contract.file(name), usecols=["t_sec"])
            self.assert_times(frame.t_sec, name)
            if name == "train" and contract.data.get("window_policy", {}).get("require_observed_endpoints", False):
                if frame.t_sec.min() != self.start or frame.t_sec.max() != self.end:
                    raise ValueError("Training timestamps do not cover the configured window endpoints")
        self.assert_times(contract.data.get("analysis", {}).get("map_times_sec", []), "density maps")
        return self
