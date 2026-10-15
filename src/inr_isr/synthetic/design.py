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


def draw_matched_patch_pattern(*, seed, first_x_km, minimum_x_km,
                               spacing_km, diameter_range_km, region_y_km,
                               edge_widths_km=(40.0, 120.0), enhancement_dex=1.0):
    """Frozen pairs: same size/context, different edges; counterbalanced order.

    Spacing must be a whole front wavelength in the saved recipe for identical
    background phase. Region order is low/high/front; random sizes and plateau
    positions are drawn once per pair. Return complete pairs beyond the margin.
    """
    from .profiles import LogPatch
    if (not np.isfinite([first_x_km, minimum_x_km, spacing_km, enhancement_dex]).all()
            or spacing_km <= 0 or minimum_x_km > first_x_km or enhancement_dex <= 0):
        raise ValueError("Invalid matched-pattern extent, spacing or enhancement")
    if len(diameter_range_km) != 2 or not 0 < diameter_range_km[0] <= diameter_range_km[1]:
        raise ValueError("Invalid diameter range")
    if len(edge_widths_km) != 2 or not 0 < edge_widths_km[0] < edge_widths_km[1]:
        raise ValueError("Need ordered sharp and gradual edge widths")
    if set(region_y_km) != {"low", "high", "front"}:
        raise ValueError("Need low/high/front position ranges")
    rng = np.random.Generator(np.random.PCG64(seed))
    pairs, x, pair_index = [], float(first_x_km), 0
    while x >= minimum_x_km:
        region = ("low", "high", "front")[pair_index % 3]
        bounds = region_y_km[region]
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] < bounds[0]:
            raise ValueError("Invalid region centre range")
        y = float(rng.uniform(*bounds))
        diameter = float(rng.uniform(*diameter_range_km))
        order = (0, 1) if pair_index % 2 == 0 else (1, 0)
        for offset, kind in enumerate(order):
            item = dict(name=f"patch_{2 * pair_index + offset:03d}",
                        pair_id=f"pair_{pair_index:03d}", region=region,
                        edge_class=("sharp", "gradual")[kind], x0_km=x - offset * spacing_km,
                        y0_km=y, half_height_diameter_km=diameter,
                        edge_width_10_90_km=float(edge_widths_km[kind]),
                        enhancement_dex=float(enhancement_dex))
            LogPatch(**item)  # Reject impossible edge/core choices at design time.
            pairs.append(item)
        x -= 2 * spacing_km
        pair_index += 1
    return pairs
