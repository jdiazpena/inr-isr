"""Draw a spatial patch pattern once; persist the returned realised list."""
import numpy as np


def draw_patch_pattern(*, seed, first_x_km, minimum_x_km, y_range_km,
                       spacing_range_km, fwhm_range_km, amplitude_range_m3):
    """Frozen upstream centres; subsequent evaluators never redraw this pattern.

    FWHM is converted to Gaussian sigma explicitly. This helper prescribes
    common downstream translation later; it models no creation/fluid dynamics.
    """
    for bounds in (y_range_km, spacing_range_km, fwhm_range_km, amplitude_range_m3):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] < bounds[0]:
            raise ValueError("Invalid design bounds")
    if min(spacing_range_km[0], fwhm_range_km[0], amplitude_range_m3[0]) <= 0:
        raise ValueError("Spacing, widths and enhancements must be positive")
    if not np.isfinite([first_x_km, minimum_x_km]).all() or minimum_x_km > first_x_km:
        raise ValueError("Invalid upstream design extent")
    rng = np.random.Generator(np.random.PCG64(seed))
    patches, x = [], float(first_x_km)
    factor = 2 * np.sqrt(2 * np.log(2))
    while x >= minimum_x_km:
        patches.append({"name": f"patch_{len(patches):03d}", "x0_km": x,
                        "y0_km": float(rng.uniform(*y_range_km)),
                        "sigma_x_km": float(rng.uniform(*fwhm_range_km) / factor),
                        "sigma_y_km": float(rng.uniform(*fwhm_range_km) / factor),
                        "amplitude_m3": float(rng.uniform(*amplitude_range_m3))})
        x -= float(rng.uniform(*spacing_range_km))
    return patches
