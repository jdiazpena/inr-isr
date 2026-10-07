"""Observation arithmetic, training-only baselines and explicit group metrics.

Tensor arithmetic is independently testable without constructing a network.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import Delaunay
import torch

from inr_isr.analysis import compute_error_metrics
from .data import exposure_points, quadrature_weights, NODES


def tensor_observation_log10(normalized_nodes, scaler):
    """[N,33] normalized point outputs → physical log10 linear-Ne average.

Inverse target scaling precedes the nonlinear observation operator. logsumexp
avoids overflow; no detach occurs, so each integration node retains its gradient.
"""
    if normalized_nodes.ndim != 2 or normalized_nodes.shape[1] != NODES:
        raise ValueError("Observation outputs must have shape [N,33]")
    physical_log = 0.5 * (normalized_nodes + 1) * (scaler["max"] - scaler["min"]) + scaler["min"]
    log_weights = torch.as_tensor(np.log(quadrature_weights()), dtype=normalized_nodes.dtype, device=normalized_nodes.device)
    return torch.logsumexp(physical_log * math.log(10) + log_weights, dim=1) / math.log(10)


def tensor_observation_mse(normalized_nodes, physical_targets, scaler):
    prediction = tensor_observation_log10(normalized_nodes, scaler)
    prediction_norm = 2 * (prediction - scaler["min"]) / (scaler["max"] - scaler["min"]) - 1
    target_norm = 2 * (physical_targets - scaler["min"]) / (scaler["max"] - scaler["min"]) - 1
    return torch.mean((prediction_norm - target_norm) ** 2)


def numpy_log_average(physical_nodes):
    physical_nodes = np.asarray(physical_nodes, dtype=np.float64)
    if physical_nodes.ndim != 2 or physical_nodes.shape[1] != NODES:
        raise ValueError("Physical node values must have shape [N,33]")
    maximum = np.max(physical_nodes, axis=1)
    return maximum + np.log10(np.sum(10 ** (physical_nodes - maximum[:, None]) * quadrature_weights(), axis=1))


class LinearBaseline:
    """Temporal linear log interpolation per beam, then spatial barycentric mix.

Only training observations define nodes. Those interval averages are explicitly
treated as midpoint nodal values; this is not deconvolution/inverse estimation.
"""
    def __init__(self, training_rows):
        if not len(training_rows) or set(training_rows.role) != {"train"}:
            raise ValueError("Baseline fitting requires training rows only")
        self.nodes = []
        xy = []
        for _, group in training_rows.groupby("beam_index", sort=True):
            group = group.sort_values("t_sec")
            if group.t_sec.duplicated().any() or len(group) < 2:
                raise ValueError("Each training beam needs distinct ordered times")
            if not np.all(group[["x_km", "y_km"]].to_numpy() == group[["x_km", "y_km"]].iloc[0].to_numpy()):
                raise ValueError("This baseline requires fixed beam locations")
            xy.append(group[["x_km", "y_km"]].iloc[0].to_numpy())
            self.nodes.append((group.t_sec.to_numpy(), group.log10_Ne.to_numpy(),
                               float(group.start_sec.min()), float(group.end_sec.max())))
        self.triangulation = Delaunay(np.asarray(xy))
        self.constant = float(training_rows.log10_Ne.mean())

    def predict(self, points, endpoint_extension=False):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Baseline queries must have shape [N,3]")
        values = np.full(len(points), np.nan)
        supported = np.zeros(len(points), dtype=bool)
        extended = np.zeros(len(points), dtype=bool)
        simplex = self.triangulation.find_simplex(points[:, :2])
        good = np.flatnonzero(simplex >= 0)
        for cell in np.unique(simplex[good]):
            indices = good[simplex[good] == cell]
            transform = self.triangulation.transform[cell]
            bary = (points[indices, :2] - transform[2]) @ transform[:2].T
            bary = np.c_[bary, 1 - bary.sum(axis=1)]
            vertex_values, vertex_support, vertex_extended = [], [], []
            for vertex in self.triangulation.simplices[cell]:
                times, logs, exposure_start, exposure_end = self.nodes[vertex]
                time = points[indices, 2]
                inside = (time >= times[0]) & (time <= times[-1])
                exception = endpoint_extension & (time >= exposure_start) & (time <= exposure_end)
                vertex_support.append(inside | exception)
                vertex_extended.append(~inside & exception)
                vertex_values.append(np.interp(time, times, logs))
            support = np.all(vertex_support, axis=0)
            output = np.sum(np.column_stack(vertex_values) * bary, axis=1)
            values[indices[support]] = output[support]
            supported[indices] = support
            extended[indices] = np.any(vertex_extended, axis=0) & support
        return values, supported, extended

    def observations(self, rows):
        nodes = exposure_points(rows)
        values, support, extended = self.predict(nodes.reshape(-1, 3), endpoint_extension=True)
        valid = support.reshape(len(rows), NODES).all(axis=1)
        prediction = numpy_log_average(values.reshape(len(rows), NODES))
        return prediction, valid, extended.reshape(len(rows), NODES).any(axis=1)


def metrics(truth_log, prediction_log, groups=None):
    """Caller supplies an explicit common supported mask; never drop rows here."""
    truth, prediction = np.asarray(truth_log, dtype=float), np.asarray(prediction_log, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 1 or not len(truth) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require aligned, nonempty finite log-density arrays")
    error = prediction - truth
    physical_truth, physical_prediction = 10 ** truth, 10 ** prediction
    if not np.isfinite(physical_prediction).all() or not np.isfinite(physical_truth).all() or np.any(physical_truth <= 0):
        raise ValueError("Invalid physical-density metric values")
    result = {"count": len(truth), "log10_pooled": compute_error_metrics(error),
              "linear_m3": compute_error_metrics(physical_prediction - physical_truth),
              "absolute_percentage_error": {"median": float(np.median(100 * np.abs(physical_prediction - physical_truth) / physical_truth)),
                                             "p90": float(np.percentile(100 * np.abs(physical_prediction - physical_truth) / physical_truth, 90))}}
    group_rows = []
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != truth.shape:
            raise ValueError("Metric groups are not aligned")
        for group in np.unique(groups):
            selected = groups == group
            entry = compute_error_metrics(error[selected])
            linear = compute_error_metrics((physical_prediction - physical_truth)[selected])
            group_rows.append({"group": group.item(), "count": int(selected.sum()), **entry,
                               **{f"linear_{key}": value for key, value in linear.items()}})
        result["log10_macro"] = {"mae": float(np.mean([r["mae"] for r in group_rows])),
                                 "rmse": float(np.sqrt(np.mean([r["mse"] for r in group_rows]))),
                                 "bias": float(np.mean([r["bias"] for r in group_rows])), "groups": len(group_rows)}
        result["linear_macro_m3"] = {"mae": float(np.mean([r["linear_mae"] for r in group_rows])),
                                     "rmse": float(np.sqrt(np.mean([r["linear_mse"] for r in group_rows]))),
                                     "bias": float(np.mean([r["linear_bias"] for r in group_rows])), "groups": len(group_rows)}
    return result, group_rows


def _width(axis, profile, peak, threshold):
    """Contiguous half-excess width; absent crossing means truncated/undefined."""
    crossings = []
    for direction in (-1, 1):
        index = peak
        while 0 <= index + direction < len(axis):
            other = index + direction
            if not np.isfinite(profile[other]):
                break
            if profile[other] <= threshold:
                if profile[index] == profile[other]:
                    break
                crossings.append(float(axis[index] + (axis[other] - axis[index]) *
                                       (threshold - profile[index]) / (profile[other] - profile[index])))
                break
            index = other
    return abs(crossings[1] - crossings[0]) if len(crossings) == 2 else None


def feature_summary(x, y, log_image, background):
    """Sampled-map peak and linear-Ne half-excess widths through its own peak.

    Grid resolution and hull truncation limit these development diagnostics.
    A nonpositive excess or a missing crossing produces an undefined width.
    """
    image = np.asarray(log_image)
    if image.shape != (len(y), len(x)) or not np.isfinite(image).any():
        raise ValueError("Feature map shape/support mismatch")
    iy, ix = np.unravel_index(np.nanargmax(image), image.shape)
    peak = float(10 ** image[iy, ix])
    excess = peak - background
    threshold = background + excess / 2
    width_x = _width(np.asarray(x), 10 ** image[iy], ix, threshold) if excess > 0 else None
    width_y = _width(np.asarray(y), 10 ** image[:, ix], iy, threshold) if excess > 0 else None
    return {"peak_x_km": float(x[ix]), "peak_y_km": float(y[iy]), "peak_ne_m3": peak,
            "excess_amplitude_m3": excess, "fwhm_x_km": width_x, "fwhm_y_km": width_y,
            "width_undefined_or_truncated": width_x is None or width_y is None}
