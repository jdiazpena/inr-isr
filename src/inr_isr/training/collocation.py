"""Optional physical beam-hull buffer; legacy nearest-beam sampling stays intact."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import ConvexHull


def collocation_definition(region="nearest_beam", buffer_km=None):
    if region not in ("nearest_beam", "beam_hull_buffer"):
        raise ValueError("Unknown collocation region")
    if region == "nearest_beam":
        if buffer_km is not None:
            raise ValueError("A buffer requires collocation_region=beam_hull_buffer")
    elif (isinstance(buffer_km, bool) or buffer_km is None
          or not np.isfinite(buffer_km) or buffer_km < 0):
        raise ValueError("Beam-hull buffer must be finite and nonnegative in km")
    return {"collocation_region": region, "collocation_buffer_km": buffer_km}


def distance_to_beam_hull(points, measured_xy):
    """Euclidean km distance to the closed convex polygon; interior distance is zero.

    Segment distances give rounded buffer corners, rather than offset-edge
    intersections that extend too far from a polygon vertex.
    """
    points = np.asarray(points, dtype=float)
    measured_xy = np.asarray(measured_xy, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or measured_xy.ndim != 2 or measured_xy.shape[1] != 2
            or not np.isfinite(measured_xy).all()):
        raise ValueError("Hull coordinates must be finite two-column arrays")
    hull = ConvexHull(measured_xy)
    vertices = measured_xy[hull.vertices]
    distance2 = np.full(len(points), np.inf)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = b - a
        fraction = np.clip((points - a) @ edge / (edge @ edge), 0., 1.)
        delta = points - (a + fraction[:, None] * edge)
        distance2 = np.minimum(distance2, np.einsum("ij,ij->i", delta, delta))
    inside = np.all(points @ hull.equations[:, :2].T + hull.equations[:, 2] <= 0, axis=1)
    return np.where(inside, 0., np.sqrt(distance2))


def _extended_axis(axis, lower, upper):
    """Keep the original lattice exactly, extending it to cover the full margin."""
    if len(axis) < 2:
        raise ValueError("Buffered collocation grid needs at least two sites per axis")
    step = float(axis[1] - axis[0])
    before = max(0, math.ceil((axis[0] - lower) / step))
    after = max(0, math.ceil((upper - axis[-1]) / step))
    return np.concatenate((axis[0] - step * np.arange(before, 0, -1), axis,
                           axis[-1] + step * np.arange(1, after + 1)))


def buffered_beam_grid(measured_xy, grid_nx, grid_ny, padding_frac, buffer_km):
    """Extend the legacy grid spacing, then mask by hull distance in physical km."""
    from training_common import make_query_grid_from_points
    collocation_definition("beam_hull_buffer", buffer_km)
    measured_xy = np.asarray(measured_xy, dtype=float)
    X, Y = make_query_grid_from_points(measured_xy[:, 0], measured_xy[:, 1],
                                       grid_nx, grid_ny, padding_frac)
    x = _extended_axis(X[0], measured_xy[:, 0].min() - buffer_km,
                       measured_xy[:, 0].max() + buffer_km)
    y = _extended_axis(Y[:, 0], measured_xy[:, 1].min() - buffer_km,
                       measured_xy[:, 1].max() + buffer_km)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    mask = (distance_to_beam_hull(points, measured_xy) <= buffer_km).reshape(X.shape)
    return X, Y, mask


def make_buffered_collocation_pool(dataset, df, grid_nx, grid_ny, padding_frac, buffer_km):
    """Use the agreed spatial polygon at the existing observation times only."""
    import torch
    from training_common import normalize_xy_t_grid_with_dataset
    xy = df[["x_km", "y_km"]].drop_duplicates().to_numpy(dtype=float)
    X, Y, mask = buffered_beam_grid(xy, grid_nx, grid_ny, padding_frac, buffer_km)
    times = np.sort(df.t_sec.unique())
    points = np.concatenate([
        normalize_xy_t_grid_with_dataset(dataset, X[mask].reshape(-1, 1),
                                         Y[mask].reshape(-1, 1), float(t)) for t in times
    ], axis=0).astype(np.float32)
    if not len(points):
        raise ValueError("No beam-buffer collocation points")
    return torch.from_numpy(points), float(buffer_km), float(mask.mean()), len(times)
