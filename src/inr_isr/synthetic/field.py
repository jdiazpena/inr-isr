"""Frozen two-region field with a translating sinusoidal front and Gaussian patches.

Coordinates are x/y in km, time in seconds and density in m^-3. Common rigid
translation is a benchmark simplification, not a plasma evolution equation.
The older synthetic_plasma generator and its defaults are preserved.
"""
from __future__ import annotations

import numpy as np

from synthetic_plasma import MovingGaussianPatch


class StructuredScene:
    """Evaluate one saved spatial pattern at a selected positive x velocity.

    ``definition`` contains realised parameters, not random distribution rules.
    No random numbers are drawn here. Truth extends by a declared time margin
    for centred exposures; queries beyond it fail instead of padding endpoints.
    """

    def __init__(self, definition: dict, speed_km_s: float):
        if definition.get("schema_version") != 1:
            raise ValueError("Unsupported structured-scene schema")
        self.definition = definition
        self.speed = float(speed_km_s)
        self.duration = float(definition["duration_sec"])
        self.margin = float(definition["truth_margin_sec"])
        self.front = {k: float(v) for k, v in definition["front"].items()}
        f = self.front
        required = ("low_ne_m3", "high_ne_m3", "center_y_km", "amplitude_km",
                    "wavelength_km", "vertical_width_10_90_km", "phase_rad")
        if set(f) != set(required) or not all(np.isfinite(f[k]) for k in required):
            raise ValueError("Front needs finite, explicitly named parameters")
        if not (0 < f["low_ne_m3"] < f["high_ne_m3"] and
                f["wavelength_km"] > 0 and f["vertical_width_10_90_km"] > 0):
            raise ValueError("Invalid density levels or front widths")
        if not (np.isfinite([self.speed, self.duration, self.margin]).all() and
                self.speed > 0 and self.duration > 0 and self.margin >= 0):
            raise ValueError("Invalid speed, duration or truth margin")
        self.patches = []
        for item in definition["patches"]:
            p = MovingGaussianPatch(**item, vx_km_s=self.speed, vy_km_s=0.0)
            if not (np.isfinite([p.amplitude_m3, p.sigma_x_km, p.sigma_y_km,
                                 p.x0_km, p.y0_km]).all() and p.amplitude_m3 > 0
                    and p.sigma_x_km > 0 and p.sigma_y_km > 0):
                raise ValueError("Patches must be finite positive enhancements")
            self.patches.append(p)
        if len({p.name for p in self.patches}) != len(self.patches):
            raise ValueError("Patch identifiers must be unique")

    def boundary_y(self, x_km, t_sec):
        f = self.front
        return f["center_y_km"] + f["amplitude_km"] * np.sin(
            2 * np.pi * (np.asarray(x_km) - self.speed * np.asarray(t_sec)) /
            f["wavelength_km"] + f["phase_rad"])

    def evaluate(self, x_km, y_km, t_sec, *, derivatives=True, render_tail_sigma=None):
        """Values and analytic first/second log-density derivatives.

        Default truth sums every Gaussian. Optional whole-component omission is
        for rendering only: omit a patch only when ALL query points lie farther
        than render_tail_sigma widths in x or y. The returned absolute density
        error bound is sum(amplitudes)*exp(-sigma^2/2); it does not change truth.
        """
        x, y, t = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64)
                                       for a in (x_km, y_km, t_sec)])
        if not all(np.isfinite(a).all() for a in (x, y, t)) or x.size == 0:
            raise ValueError("Queries must be nonempty and finite")
        if np.min(t) < -self.margin or np.max(t) > self.duration + self.margin:
            raise ValueError("Query exceeds the saved analytic truth time margin")
        if render_tail_sigma is not None and (derivatives or render_tail_sigma < 1):
            raise ValueError("Tail omission is a values-only rendering option")
        f = self.front
        u = x - self.speed * t
        k = 2 * np.pi / f["wavelength_km"]
        theta = k * u + f["phase_rad"]
        ell = f["vertical_width_10_90_km"] / (2 * np.arctanh(0.8))
        q = (y - self.boundary_y(x, t)) / ell
        z = np.tanh(q)
        fraction = 0.5 * (1 + z)
        contrast = f["high_ne_m3"] - f["low_ne_m3"]
        ne = f["low_ne_m3"] + contrast * fraction
        patch_ne = np.zeros_like(ne)
        if derivatives:
            qx = -f["amplitude_km"] * k * np.cos(theta) / ell
            qy = 1 / ell
            qxx = f["amplitude_km"] * k**2 * np.sin(theta) / ell
            s1 = 0.5 * (1 - z**2)
            s2 = -(1 - z**2) * z
            nx, ny = contrast * s1 * qx, contrast * s1 * qy
            nxx = contrast * (s2 * qx**2 + s1 * qxx)
            nxy = contrast * s2 * qx * qy
            nyy = contrast * s2 * qy**2
        bounds = (float(u.min()), float(u.max()), float(y.min()), float(y.max()))
        omitted_bound = 0.0
        for p in self.patches:
            if render_tail_sigma is not None:
                sx, sy = render_tail_sigma * p.sigma_x_km, render_tail_sigma * p.sigma_y_km
                if (p.x0_km < bounds[0] - sx or p.x0_km > bounds[1] + sx or
                        p.y0_km < bounds[2] - sy or p.y0_km > bounds[3] + sy):
                    omitted_bound += p.amplitude_m3 * np.exp(-0.5 * render_tail_sigma**2)
                    continue
            values = p.evaluate_delta_and_derivatives(x, y, t)
            delta = values["delta_ne"]
            ne = ne + delta
            patch_ne = patch_ne + delta
            if derivatives:
                nx += values["d_delta_dx_km"]
                ny += values["d_delta_dy_km"]
                dx, dy = u - p.x0_km, y - p.y0_km
                nxx += delta * ((dx / p.sigma_x_km**2)**2 - 1 / p.sigma_x_km**2)
                nxy += delta * dx * dy / (p.sigma_x_km**2 * p.sigma_y_km**2)
                nyy += delta * ((dy / p.sigma_y_km**2)**2 - 1 / p.sigma_y_km**2)
        out = {"Ne": ne, "log10_Ne": np.log10(ne), "high_region_fraction": fraction,
               "patch_delta_ne_m3": patch_ne, "render_density_error_bound_m3": omitted_bound}
        if derivatives:
            nt = -self.speed * nx
            physical = {"x_km": nx, "y_km": ny, "t_sec": nt}
            ln10 = np.log(10.0)
            for label, first in physical.items():
                out[f"true_dlog10Ne_d{label}"] = first / (ne * ln10)
            second = {"xx_km2": (nxx, nx, nx), "xy_km2": (nxy, nx, ny),
                      "yy_km2": (nyy, ny, ny),
                      "tt_sec2": (self.speed**2 * nxx, nt, nt)}
            for label, (curvature, a, b) in second.items():
                out[f"true_d2log10Ne_d{label}"] = (curvature / ne - (a / ne) * (b / ne)) / ln10
        return out


def scene_from_definition(definition, speed_km_s):
    """Explicit recipe-version dispatch; keep schema1 numerical behaviour intact."""
    if definition.get("schema_version") == 2:
        from .log_field import LogStructuredScene
        return LogStructuredScene(definition, speed_km_s)
    return StructuredScene(definition, speed_km_s)


def average_scene(scene, x_km, y_km, t_sec, exposure_sec, *, max_spacing_sec=0.1):
    """Centred trapezoidal average of underlying LINEAR density, then log10.

    Includes exposure endpoints. The analytic scene supplies the margins, with
    no constant padding or averaging of saved observations. Chunked evaluation
    bounds working memory; cadence and integration spacing are independent.
    First derivatives of the average are obtained by integrating Ne derivatives.
    """
    if not np.isfinite(exposure_sec) or exposure_sec < 0:
        raise ValueError("Exposure must be finite and nonnegative")
    if exposure_sec == 0:
        return scene.evaluate(x_km, y_km, t_sec)
    if not np.isfinite(max_spacing_sec) or max_spacing_sec <= 0:
        raise ValueError("Integration spacing must be finite and positive")
    x, y, t = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64)
                                   for a in (x_km, y_km, t_sec)])
    count = int(np.ceil(exposure_sec / max_spacing_sec)) + 1
    offsets = np.linspace(-exposure_sec / 2, exposure_sec / 2, count)
    keys = ("true_dlog10Ne_dx_km", "true_dlog10Ne_dy_km", "true_dlog10Ne_dt_sec")
    ne = np.empty(x.size)
    first = {key: np.empty(x.size) for key in keys}
    for start in range(0, x.size, 64):
        stop = min(start + 64, x.size)
        vals = scene.evaluate(x.ravel()[start:stop, None], y.ravel()[start:stop, None],
                              t.ravel()[start:stop, None] + offsets)
        ne[start:stop] = np.trapezoid(vals["Ne"], offsets, axis=-1) / exposure_sec
        for key in keys:
            linear_derivative = vals[key] * vals["Ne"] * np.log(10.0)
            first[key][start:stop] = np.trapezoid(linear_derivative, offsets, axis=-1) / exposure_sec
    out = {"Ne": ne.reshape(x.shape), "log10_Ne": np.log10(ne).reshape(x.shape)}
    out.update({key: (values / (ne * np.log(10.0))).reshape(x.shape)
                for key, values in first.items()})
    return out
