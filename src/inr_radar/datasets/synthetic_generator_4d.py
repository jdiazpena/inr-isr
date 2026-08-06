# -*- coding: utf-8 -*-
"""
synthetic_generator_4d.py

Synthetic 3D continuous space + 1D time plasma generator for 4D INR experiments.

Coordinates:
    x_km: East-West horizontal coordinate [km]
    y_km: North-South horizontal coordinate [km]
    z_km: Vertical altitude coordinate [km] (100 km <= z_km <= 500 km)
    t_sec: Time [s]

Target:
    log10_Ne = log10(Ne)

Combines horizontal Gaussian convection in (x, y, t) with a realistic vertical F2 altitude profile
centered at z_peak ~ 300 km with scale height H = 50 km.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class MovingGaussianPatch4D:
    """
    4D Moving Gaussian patch with horizontal convection (x, y, t) and vertical envelope z.
    """

    name: str = "patch_4d_1"
    amplitude_m3: float = 1.0e12
    sigma_x_km: float = 45.0
    sigma_y_km: float = 45.0
    x0_km: float = -180.0
    y0_km: float = 0.0
    vx_km_s: float = 0.1
    vy_km_s: float = 0.0
    z_peak_km: float = 300.0
    scale_height_km: float = 50.0

    def center(self, t_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t_sec = np.asarray(t_sec, dtype=np.float64)
        xc = self.x0_km + self.vx_km_s * t_sec
        yc = self.y0_km + self.vy_km_s * t_sec
        return xc, yc

    def evaluate_delta_and_derivatives(
        self,
        x_km: np.ndarray,
        y_km: np.ndarray,
        z_km: np.ndarray,
        t_sec: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Evaluate patch delta Ne and physical coordinate derivatives.
        """
        x = np.asarray(x_km, dtype=np.float64)
        y = np.asarray(y_km, dtype=np.float64)
        z = np.asarray(z_km, dtype=np.float64)
        t = np.asarray(t_sec, dtype=np.float64)

        xc, yc = self.center(t)
        dx = x - xc
        dy = y - yc

        sx2 = self.sigma_x_km ** 2
        sy2 = self.sigma_y_km ** 2

        # Horizontal Gaussian envelope
        exp_h = np.exp(-0.5 * ((dx ** 2) / sx2 + (dy ** 2) / sy2))

        # Vertical Chapman-like F2 profile envelope
        z_norm = (z - self.z_peak_km) / self.scale_height_km
        exp_v = np.exp(1.0 - z_norm - np.exp(-z_norm))

        delta_ne = self.amplitude_m3 * exp_h * exp_v

        # Spatial and temporal derivatives of delta_ne
        d_delta_dx = delta_ne * (-dx / sx2)
        d_delta_dy = delta_ne * (-dy / sy2)
        d_delta_dz = delta_ne * ((-1.0 + np.exp(-z_norm)) / self.scale_height_km)
        d_delta_dt = delta_ne * (dx * self.vx_km_s / sx2 + dy * self.vy_km_s / sy2)

        return {
            "delta_ne": delta_ne,
            "d_delta_dx_km": d_delta_dx,
            "d_delta_dy_km": d_delta_dy,
            "d_delta_dz_km": d_delta_dz,
            "d_delta_dt_sec": d_delta_dt,
            "center_x_km": xc,
            "center_y_km": yc,
        }


def vertical_f2_profile(
    z_km: np.ndarray,
    z_peak_km: float = 300.0,
    scale_height_km: float = 50.0,
) -> np.ndarray:
    """
    Chapman F2 layer vertical profile normalized to 1.0 at peak altitude.
    """
    z = np.asarray(z_km, dtype=np.float64)
    z_norm = (z - z_peak_km) / scale_height_km
    return np.exp(1.0 - z_norm - np.exp(-z_norm))


def evaluate_synthetic_plasma_4d(
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_km: np.ndarray,
    t_sec: np.ndarray,
    background_ne_m3: float = 2.0e11,
    patches: list[MovingGaussianPatch4D] | None = None,
    z_peak_km: float = 300.0,
    scale_height_km: float = 50.0,
    min_ne_m3: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Evaluate total Ne(x, y, z, t), log10(Ne), and analytical derivatives.
    """
    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    z = np.asarray(z_km, dtype=np.float64)
    t = np.asarray(t_sec, dtype=np.float64)

    target_shape = np.broadcast_shapes(x.shape, y.shape, z.shape, t.shape)

    x_b = np.broadcast_to(x, target_shape)
    y_b = np.broadcast_to(y, target_shape)
    z_b = np.broadcast_to(z, target_shape)
    t_b = np.broadcast_to(t, target_shape)

    # Background electron density profile
    v_bg = vertical_f2_profile(z_b, z_peak_km=z_peak_km, scale_height_km=scale_height_km)
    ne = background_ne_m3 * v_bg

    z_norm_bg = (z_b - z_peak_km) / scale_height_km
    d_ne_dx = np.zeros_like(ne)
    d_ne_dy = np.zeros_like(ne)
    d_ne_dz = background_ne_m3 * v_bg * ((-1.0 + np.exp(-z_norm_bg)) / scale_height_km)
    d_ne_dt = np.zeros_like(ne)

    if patches is not None:
        for patch in patches:
            res = patch.evaluate_delta_and_derivatives(x_b, y_b, z_b, t_b)
            ne = ne + res["delta_ne"]
            d_ne_dx = d_ne_dx + res["d_delta_dx_km"]
            d_ne_dy = d_ne_dy + res["d_delta_dy_km"]
            d_ne_dz = d_ne_dz + res["d_delta_dz_km"]
            d_ne_dt = d_ne_dt + res["d_delta_dt_sec"]

    ne_clipped = np.clip(ne, float(min_ne_m3), None)
    log10_ne = np.log10(ne_clipped)

    inv_ln10_ne = 1.0 / (np.log(10.0) * ne_clipped)

    dlog10_dx = d_ne_dx * inv_ln10_ne
    dlog10_dy = d_ne_dy * inv_ln10_ne
    dlog10_dz = d_ne_dz * inv_ln10_ne
    dlog10_dt = d_ne_dt * inv_ln10_ne

    return {
        "x_km": x_b,
        "y_km": y_b,
        "z_km": z_b,
        "t_sec": t_b,
        "Ne": ne_clipped,
        "log10_Ne": log10_ne,
        "dlog10Ne_dx_km": dlog10_dx,
        "dlog10Ne_dy_km": dlog10_dy,
        "dlog10Ne_dz_km": dlog10_dz,
        "dlog10Ne_dt_sec": dlog10_dt,
    }


def make_observation_geometry_4d(
    n_beams: int = 42,
    z_min_km: float = 100.0,
    z_max_km: float = 500.0,
    n_ranges: int = 25,
    n_times: int = 10,
    duration_sec: float = 3600.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic sparse radar beam sampling geometry across 100 to 500 km altitudes.
    """
    rng = np.random.default_rng(seed)

    # Beam angles (azimuth and elevation) mimicking PFISR/AMISR beam cluster
    az_deg = rng.uniform(0.0, 360.0, size=n_beams)
    el_deg = rng.uniform(30.0, 85.0, size=n_beams)

    times = np.linspace(0.0, duration_sec, n_times)
    altitudes = np.linspace(z_min_km, z_max_km, n_ranges)

    rows = []
    for b_idx in range(n_beams):
        az = az_deg[b_idx]
        el = el_deg[b_idx]
        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(el)

        for r_idx, z in enumerate(altitudes):
            # Compute range and horizontal x, y coordinates along beam
            r_km = z / np.sin(el_rad)
            x = r_km * np.cos(el_rad) * np.sin(az_rad)
            y = r_km * np.cos(el_rad) * np.cos(az_rad)

            for t_idx, t in enumerate(times):
                rows.append({
                    "beam_index": b_idx,
                    "beamcode": float(1000 + b_idx),
                    "az_deg": az,
                    "el_deg": el,
                    "range_index": r_idx,
                    "range_km": r_km,
                    "x_km": x,
                    "y_km": y,
                    "z_km": z,
                    "altitude_km": z,
                    "time_index": t_idx,
                    "t_sec": t,
                })

    df_geom = pd.DataFrame(rows)
    return df_geom


def generate_synthetic_beam_dataset_4d(
    background_ne_m3: float = 2.0e11,
    patches: list[MovingGaussianPatch4D] | None = None,
    n_beams: int = 42,
    z_min_km: float = 100.0,
    z_max_km: float = 500.0,
    n_ranges: int = 25,
    n_times: int = 10,
    duration_sec: float = 3600.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 4D plasma observation dataset (x_km, y_km, z_km, t_sec) -> log10_Ne.
    """
    if patches is None:
        patches = [
            MovingGaussianPatch4D(
                name="auroral_patch_1",
                amplitude_m3=1.5e12,
                sigma_x_km=50.0,
                sigma_y_km=50.0,
                x0_km=-150.0,
                y0_km=0.0,
                vx_km_s=0.1,
                vy_km_s=0.05,
                z_peak_km=300.0,
                scale_height_km=50.0,
            )
        ]

    df_geom = make_observation_geometry_4d(
        n_beams=n_beams,
        z_min_km=z_min_km,
        z_max_km=z_max_km,
        n_ranges=n_ranges,
        n_times=n_times,
        duration_sec=duration_sec,
        seed=seed,
    )

    res = evaluate_synthetic_plasma_4d(
        x_km=df_geom["x_km"].to_numpy(),
        y_km=df_geom["y_km"].to_numpy(),
        z_km=df_geom["z_km"].to_numpy(),
        t_sec=df_geom["t_sec"].to_numpy(),
        background_ne_m3=background_ne_m3,
        patches=patches,
    )

    for k in [
        "Ne",
        "log10_Ne",
        "dlog10Ne_dx_km",
        "dlog10Ne_dy_km",
        "dlog10Ne_dz_km",
        "dlog10Ne_dt_sec",
    ]:
        df_geom[k] = res[k]

    return df_geom
