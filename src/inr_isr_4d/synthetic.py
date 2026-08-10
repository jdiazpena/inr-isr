"""Named analytic 4D truths and integration-aware synthetic radar observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import FieldBundle4D


SyntheticVariant = Literal["gaussian_3d", "chapman_f2_reference"]
ObservationMode = Literal["instantaneous", "integration_averaged"]


@dataclass(frozen=True)
class MovingFeature4D:
    amplitude_m3: float = 5.0e11
    sigma_x_km: float = 50.0
    sigma_y_km: float = 50.0
    sigma_z_km: float = 35.0
    x0_km: float = -80.0
    y0_km: float = 0.0
    z0_km: float = 300.0
    velocity_x_km_s: float = 0.25
    velocity_y_km_s: float = 0.10
    velocity_z_km_s: float = 0.0

    def validate(self) -> None:
        values = np.asarray(list(self.__dict__.values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Synthetic feature parameters must be finite.")
        if min(self.sigma_x_km, self.sigma_y_km, self.sigma_z_km) <= 0:
            raise ValueError("Synthetic feature scales must be positive.")


@dataclass(frozen=True)
class SyntheticFieldConfig:
    variant: SyntheticVariant = "gaussian_3d"
    background_ne_m3: float = 1.0e11
    background_peak_km: float = 300.0
    background_scale_height_km: float = 50.0
    minimum_ne_m3: float = 1.0
    feature: MovingFeature4D = MovingFeature4D()

    def validate(self) -> None:
        if self.variant not in {"gaussian_3d", "chapman_f2_reference"}:
            raise ValueError(f"Unknown synthetic variant: {self.variant}")
        values = np.asarray(
            [
                self.background_ne_m3,
                self.background_peak_km,
                self.background_scale_height_km,
                self.minimum_ne_m3,
            ]
        )
        if not np.all(np.isfinite(values)) or min(
            self.background_ne_m3, self.background_scale_height_km, self.minimum_ne_m3
        ) <= 0:
            raise ValueError("Synthetic field background parameters are invalid.")
        self.feature.validate()


@dataclass(frozen=True)
class SyntheticObservationConfig:
    mode: ObservationMode = "instantaneous"
    integration_duration_sec: float = 0.0
    integration_samples: int = 1
    n_beams: int = 12
    n_ranges: int = 8
    n_times: int = 8
    duration_sec: float = 900.0
    altitude_min_km: float = 120.0
    altitude_max_km: float = 500.0
    seed: int = 0

    def validate(self) -> None:
        if self.mode not in {"instantaneous", "integration_averaged"}:
            raise ValueError(f"Unknown observation mode: {self.mode}")
        if min(self.n_beams, self.n_ranges, self.n_times) <= 0 or self.seed < 0:
            raise ValueError("Synthetic geometry counts must be positive and seed non-negative.")
        if self.n_times < 2 or self.duration_sec <= 0:
            raise ValueError("Synthetic observations require at least two times and positive duration.")
        if self.altitude_max_km <= self.altitude_min_km:
            raise ValueError("Synthetic altitude bounds are invalid.")
        if self.mode == "instantaneous":
            if self.integration_duration_sec != 0 or self.integration_samples != 1:
                raise ValueError("Instantaneous observations require duration zero and one sample.")
        elif self.integration_duration_sec <= 0 or self.integration_samples < 2:
            raise ValueError("Integrated observations require positive duration and at least two samples.")


@dataclass(frozen=True)
class SyntheticCase4D:
    bundle: FieldBundle4D
    instantaneous_log10_ne: np.ndarray
    integration_duration_sec: np.ndarray
    integration_start_sec: np.ndarray
    integration_end_sec: np.ndarray
    field_config: SyntheticFieldConfig
    observation_config: SyntheticObservationConfig


def _chapman(z_km: np.ndarray, peak_km: float, scale_height_km: float) -> tuple[np.ndarray, np.ndarray]:
    normalized = (z_km - peak_km) / scale_height_km
    profile = np.exp(1.0 - normalized - np.exp(-normalized))
    derivative = profile * (-1.0 + np.exp(-normalized)) / scale_height_km
    return profile, derivative


def evaluate_synthetic_truth(
    coordinates: np.ndarray,
    field_config: SyntheticFieldConfig,
) -> dict[str, np.ndarray]:
    """Evaluate linear/log density and analytic first physical derivatives."""

    field_config.validate()
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or not np.all(np.isfinite(coordinates)):
        raise ValueError("Synthetic coordinates must be a finite [N, 4] array.")
    x, y, z, t = (coordinates[:, index] for index in range(4))
    feature = field_config.feature
    center_x = feature.x0_km + feature.velocity_x_km_s * t
    center_y = feature.y0_km + feature.velocity_y_km_s * t
    center_z = feature.z0_km + feature.velocity_z_km_s * t
    dx, dy, dz = x - center_x, y - center_y, z - center_z
    horizontal = np.exp(
        -0.5 * ((dx / feature.sigma_x_km) ** 2 + (dy / feature.sigma_y_km) ** 2)
    )

    if field_config.variant == "gaussian_3d":
        vertical = np.exp(-0.5 * (dz / feature.sigma_z_km) ** 2)
        vertical_derivative = vertical * (-dz / feature.sigma_z_km**2)
        background = np.full(len(coordinates), field_config.background_ne_m3)
        background_dz = np.zeros(len(coordinates))
    else:
        vertical, vertical_derivative = _chapman(
            z, feature.z0_km, feature.sigma_z_km
        )
        background_profile, background_profile_derivative = _chapman(
            z, field_config.background_peak_km, field_config.background_scale_height_km
        )
        background = field_config.background_ne_m3 * background_profile
        background_dz = field_config.background_ne_m3 * background_profile_derivative

    delta = feature.amplitude_m3 * horizontal * vertical
    d_delta_dx = delta * (-dx / feature.sigma_x_km**2)
    d_delta_dy = delta * (-dy / feature.sigma_y_km**2)
    d_delta_dz = feature.amplitude_m3 * horizontal * vertical_derivative
    d_delta_dt = delta * (
        dx * feature.velocity_x_km_s / feature.sigma_x_km**2
        + dy * feature.velocity_y_km_s / feature.sigma_y_km**2
    )
    if field_config.variant == "gaussian_3d":
        d_delta_dt += delta * dz * feature.velocity_z_km_s / feature.sigma_z_km**2

    ne = background + delta
    clipped = ne < field_config.minimum_ne_m3
    ne = np.maximum(ne, field_config.minimum_ne_m3)
    derivatives_linear = {
        "x": np.where(clipped, 0.0, d_delta_dx),
        "y": np.where(clipped, 0.0, d_delta_dy),
        "z": np.where(clipped, 0.0, background_dz + d_delta_dz),
        "t": np.where(clipped, 0.0, d_delta_dt),
    }
    inverse = 1.0 / (np.log(10.0) * ne)
    result = {"Ne": ne, "log10_Ne": np.log10(ne)}
    for name, values in derivatives_linear.items():
        result[f"dNe_d{name}"] = values
        result[f"dlog10Ne_d{name}"] = values * inverse
    return result


def evaluate_observation_target(
    coordinates: np.ndarray,
    field_config: SyntheticFieldConfig,
    *,
    mode: ObservationMode,
    integration_duration_sec: float,
    integration_samples: int,
) -> dict[str, np.ndarray]:
    """Evaluate midpoint truth and the declared observation-product target."""

    coordinates = np.asarray(coordinates, dtype=float)
    midpoint = evaluate_synthetic_truth(coordinates, field_config)
    if mode == "instantaneous":
        if integration_duration_sec != 0 or integration_samples != 1:
            raise ValueError("Instantaneous evaluation requires duration zero and one sample.")
        return {
            **midpoint,
            "instantaneous_Ne": midpoint["Ne"],
            "instantaneous_log10_Ne": midpoint["log10_Ne"],
        }
    if mode != "integration_averaged" or integration_duration_sec <= 0 or integration_samples < 2:
        raise ValueError("Integration-averaged evaluation parameters are invalid.")

    offsets = np.linspace(-0.5 * integration_duration_sec, 0.5 * integration_duration_sec, integration_samples)
    weights = np.ones(integration_samples)
    weights[[0, -1]] = 0.5
    weights /= weights.sum()
    accumulated = {name: np.zeros(len(coordinates)) for name in ("Ne", "dNe_dx", "dNe_dy", "dNe_dz", "dNe_dt")}
    for offset, weight in zip(offsets, weights):
        shifted = coordinates.copy()
        shifted[:, 3] += offset
        values = evaluate_synthetic_truth(shifted, field_config)
        for name in accumulated:
            accumulated[name] += weight * values[name]
    ne = accumulated["Ne"]
    result = {
        "Ne": ne,
        "log10_Ne": np.log10(ne),
        "instantaneous_Ne": midpoint["Ne"],
        "instantaneous_log10_Ne": midpoint["log10_Ne"],
    }
    inverse = 1.0 / (np.log(10.0) * ne)
    for name in ("x", "y", "z", "t"):
        result[f"dNe_d{name}"] = accumulated[f"dNe_d{name}"]
        result[f"dlog10Ne_d{name}"] = accumulated[f"dNe_d{name}"] * inverse
    return result


def make_radar_geometry(config: SyntheticObservationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config.validate()
    rng = np.random.default_rng(config.seed)
    azimuth = np.deg2rad(rng.uniform(0.0, 360.0, config.n_beams))
    elevation = np.deg2rad(rng.uniform(35.0, 85.0, config.n_beams))
    altitudes = np.linspace(config.altitude_min_km, config.altitude_max_km, config.n_ranges)
    times = np.linspace(0.0, config.duration_sec, config.n_times)
    coordinates = []
    beams = []
    time_ids = []
    for time_index, time in enumerate(times):
        for beam in range(config.n_beams):
            slant_range = altitudes / np.sin(elevation[beam])
            horizontal = slant_range * np.cos(elevation[beam])
            x = horizontal * np.sin(azimuth[beam])
            y = horizontal * np.cos(azimuth[beam])
            for x_value, y_value, altitude in zip(x, y, altitudes):
                coordinates.append([x_value, y_value, altitude, time])
                beams.append(1000 + beam)
                time_ids.append(time_index)
    return np.asarray(coordinates), np.asarray(beams), np.asarray(time_ids)


def generate_synthetic_case(
    field_config: SyntheticFieldConfig,
    observation_config: SyntheticObservationConfig,
) -> SyntheticCase4D:
    field_config.validate()
    observation_config.validate()
    coordinates, beams, time_ids = make_radar_geometry(observation_config)
    values = evaluate_observation_target(
        coordinates,
        field_config,
        mode=observation_config.mode,
        integration_duration_sec=observation_config.integration_duration_sec,
        integration_samples=observation_config.integration_samples,
    )
    duration = np.full(len(coordinates), observation_config.integration_duration_sec)
    bundle = FieldBundle4D(
        coordinates=coordinates,
        targets=values["log10_Ne"][:, None],
        beam_ids=beams,
        time_ids=time_ids,
        group_ids=np.array([f"{beam}:{time}" for beam, time in zip(beams, time_ids)]),
        metadata={
            "source": "analytic synthetic truth",
            "variant": field_config.variant,
            "observation_mode": observation_config.mode,
            "integration_quadrature": "uniform composite trapezoidal rule in linear Ne",
            "integration_samples": observation_config.integration_samples,
        },
    )
    return SyntheticCase4D(
        bundle=bundle,
        instantaneous_log10_ne=values["instantaneous_log10_Ne"][:, None],
        integration_duration_sec=duration,
        integration_start_sec=coordinates[:, 3] - 0.5 * duration,
        integration_end_sec=coordinates[:, 3] + 0.5 * duration,
        field_config=field_config,
        observation_config=observation_config,
    )


def independent_truth_points(
    field_config: SyntheticFieldConfig,
    *,
    count: int,
    bounds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if count <= 0 or seed < 0:
        raise ValueError("Independent truth count must be positive and seed non-negative.")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Truth bounds must have shape [4, 2] with positive spans.")
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, 4))
    return coordinates, evaluate_synthetic_truth(coordinates, field_config)
