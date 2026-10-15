"""Schema2 frozen log-density field with independently controlled patch edges."""
import numpy as np
from .profiles import LogPatch, RAMP_10_90_FRACTION, smooth_ramp


class LogStructuredScene:
    """Prescribed common translation, with analytic log-density derivatives.

    The field is defined in log-density; physical density is supplied for the
    existing dataset/averaging interfaces. This is a reconstruction benchmark,
    not a fluid simulation. Schema1 remains implemented by StructuredScene.
    """
    def __init__(self, definition, speed_km_s):
        if definition.get("schema_version") != 2:
            raise ValueError("Expected structured-scene schema2")
        self.definition = definition
        self.speed = float(speed_km_s)
        self.duration = float(definition["duration_sec"])
        self.margin = float(definition["truth_margin_sec"])
        self.front = {k: float(v) for k, v in definition["front"].items()}
        required = {"low_log10_ne", "high_log10_ne", "center_y_km", "amplitude_km",
                    "wavelength_km", "vertical_width_10_90_km", "phase_rad"}
        f = self.front
        if set(f) != required or not np.isfinite(list(f.values())).all():
            raise ValueError("Front needs finite explicitly named log-density parameters")
        if not (f["low_log10_ne"] < f["high_log10_ne"] and
                f["wavelength_km"] > 0 and f["vertical_width_10_90_km"] > 0):
            raise ValueError("Invalid log-density levels or front widths")
        if not (np.isfinite([self.speed, self.duration, self.margin]).all() and
                self.speed > 0 and self.duration > 0 and self.margin >= 0):
            raise ValueError("Invalid speed, duration or truth margin")
        self.patches = [LogPatch(**p) for p in definition["patches"]]
        if len({p.name for p in self.patches}) != len(self.patches):
            raise ValueError("Patch identifiers must be unique")
        # Nonoverlap ensures the declared one-patch density ceiling is respected.
        for i, p in enumerate(self.patches):
            for q in self.patches[:i]:
                if np.hypot(p.x0_km - q.x0_km, p.y0_km - q.y0_km) < p.outer_radius_km + q.outer_radius_km:
                    raise ValueError("This recipe requires nonoverlapping patch supports")

    def boundary_y(self, x_km, t_sec):
        f = self.front
        return f["center_y_km"] + f["amplitude_km"] * np.sin(
            2 * np.pi * (np.asarray(x_km) - self.speed * np.asarray(t_sec)) /
            f["wavelength_km"] + f["phase_rad"])

    def evaluate(self, x_km, y_km, t_sec, *, derivatives=True, render_tail_sigma=None):
        x, y, t = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64)
                                       for a in (x_km, y_km, t_sec)])
        if not x.size or not all(np.isfinite(a).all() for a in (x, y, t)):
            raise ValueError("Queries must be finite and nonempty")
        if t.min() < -self.margin or t.max() > self.duration + self.margin:
            raise ValueError("Query exceeds saved analytic truth margins")
        if render_tail_sigma is not None and (derivatives or not np.isfinite(render_tail_sigma) or render_tail_sigma < 1):
            raise ValueError("Rendering shortcut requires values only and a finite positive width")
        u = x - self.speed * t
        f = self.front
        k = 2 * np.pi / f["wavelength_km"]
        theta = k * u + f["phase_rad"]
        width = f["vertical_width_10_90_km"] / RAMP_10_90_FRACTION
        z = (y - self.boundary_y(x, t)) / width + 0.5
        fraction, s1, s2 = smooth_ramp(z)
        contrast = f["high_log10_ne"] - f["low_log10_ne"]
        background = f["low_log10_ne"] + contrast * fraction
        log_ne = background.copy()
        enhancement = np.zeros_like(log_ne)
        if derivatives:
            zx = -f["amplitude_km"] * k * np.cos(theta) / width
            zy = 1 / width
            zxx = f["amplitude_km"] * k**2 * np.sin(theta) / width
            lx, ly = contrast * s1 * zx, contrast * s1 * zy
            lxx = contrast * (s2 * zx**2 + s1 * zxx)
            lxy, lyy = contrast * s2 * zx * zy, contrast * s2 * zy**2
        bounds = (float(u.min()), float(u.max()), float(y.min()), float(y.max()))
        for p in self.patches:
            radius = p.outer_radius_km
            if (p.x0_km < bounds[0] - radius or p.x0_km > bounds[1] + radius or
                    p.y0_km < bounds[2] - radius or p.y0_km > bounds[3] + radius):
                continue  # Exact compact support, not approximate tail truncation.
            values = p.evaluate(u - p.x0_km, y - p.y0_km, derivatives=derivatives)
            log_ne += values["enhancement_dex"]
            enhancement += values["enhancement_dex"]
            if derivatives:
                lx += values["x"]; ly += values["y"]
                lxx += values["xx"]; lxy += values["xy"]; lyy += values["yy"]
        with np.errstate(over="raise", invalid="raise"):
            ne = np.power(10.0, log_ne)
        if not np.isfinite(ne).all() or (ne <= 0).any():
            raise ValueError("Log-density recipe exceeds the finite positive density range")
        out = {"Ne": ne, "log10_Ne": log_ne, "background_log10_Ne": background,
               "high_region_fraction": fraction, "patch_enhancement_dex": enhancement,
               "patch_delta_ne_m3": ne - np.power(10.0, background),
               "render_density_error_bound_m3": 0.0}
        if derivatives:
            out.update(true_dlog10Ne_dx_km=lx, true_dlog10Ne_dy_km=ly,
                       true_dlog10Ne_dt_sec=-self.speed * lx,
                       true_d2log10Ne_dxx_km2=lxx, true_d2log10Ne_dxy_km2=lxy,
                       true_d2log10Ne_dyy_km2=lyy, true_d2log10Ne_dtt_sec2=self.speed**2 * lxx)
        return out
