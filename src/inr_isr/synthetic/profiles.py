"""Finite smooth ramps and circular patches with independent size/edge width."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq


def smooth_ramp(z):
    """C2 quintic rise on [0,1], constant outside; return value and derivatives."""
    z = np.asarray(z, dtype=np.float64)
    q = np.clip(z, 0.0, 1.0)
    value = q**3 * (10 + q * (-15 + 6 * q))
    active = (z > 0) & (z < 1)
    first = np.where(active, 30 * q**2 * (1 - q)**2, 0.0)
    second = np.where(active, 60 * q * (1 - q) * (1 - 2 * q), 0.0)
    return value, first, second


RAMP_10_90_FRACTION = 1 - 2 * brentq(
    lambda z: float(smooth_ramp(z)[0]) - 0.1, 0.0, 0.5, xtol=1e-14)


@dataclass(frozen=True)
class LogPatch:
    name: str
    pair_id: str
    region: str
    edge_class: str
    x0_km: float
    y0_km: float
    half_height_diameter_km: float
    edge_width_10_90_km: float
    enhancement_dex: float

    def __post_init__(self):
        vals = [self.x0_km, self.y0_km, self.half_height_diameter_km,
                self.edge_width_10_90_km, self.enhancement_dex]
        if not np.isfinite(vals).all() or min(vals[2:]) <= 0:
            raise ValueError("Patch positions must be finite; size, width and enhancement positive")
        if self.inner_radius_km <= 0:
            raise ValueError("Patch size must leave a flat core with the selected edge width")
        if not self.name or not self.pair_id or self.region not in ("low", "high", "front"):
            raise ValueError("Patch needs an identifier, pair and known regional context")
        if self.edge_class not in ("sharp", "gradual"):
            raise ValueError("Unknown patch edge class")

    @property
    def ramp_width_km(self):
        return self.edge_width_10_90_km / RAMP_10_90_FRACTION

    @property
    def inner_radius_km(self):
        return self.half_height_diameter_km / 2 - self.ramp_width_km / 2

    @property
    def outer_radius_km(self):
        return self.half_height_diameter_km / 2 + self.ramp_width_km / 2

    def evaluate(self, dx, dy, *, derivatives=True):
        """Log enhancement and physical-coordinate gradient/Hessian.

        Derivatives are evaluated only in the annulus, which excludes the
        radial origin. The core and exterior are constant with zero derivatives.
        """
        dx, dy = np.broadcast_arrays(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float))
        r = np.hypot(dx, dy)
        z = (r - self.inner_radius_km) / self.ramp_width_km
        value, first, second = smooth_ramp(z)
        out = {"enhancement_dex": self.enhancement_dex * (1 - value)}
        if derivatives:
            mask = (r > self.inner_radius_km) & (r < self.outer_radius_km)
            inv = np.divide(1.0, r, out=np.zeros_like(r), where=mask)
            ex, ey = dx * inv, dy * inv
            dr = -self.enhancement_dex * first / self.ramp_width_km
            drr = -self.enhancement_dex * second / self.ramp_width_km**2
            radial = dr * inv
            out.update(x=dr * ex, y=dr * ey,
                       xx=(drr - radial) * ex**2 + radial,
                       xy=(drr - radial) * ex * ey,
                       yy=(drr - radial) * ey**2 + radial)
        return out
