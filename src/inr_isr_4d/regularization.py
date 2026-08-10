"""Named 4D derivative priors with explicit coordinate conventions.

These losses are smoothness/curvature priors.  They are not plasma equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional


DerivativeMode = Literal[
    "legacy_diagonal_4d", "anisotropic_huber_4d", "spatial_hessian_3d"
]


@dataclass(frozen=True)
class PhysicalCoordinateScale:
    """Physical spans corresponding to normalized coordinate interval ``[-1, 1]``.

    Units are kilometres for x/y/z and seconds for time.  A second derivative in
    physical coordinates equals its normalized-coordinate counterpart multiplied
    by ``(2 / span) ** 2`` for each differentiated coordinate.
    """

    x_km: float
    y_km: float
    z_km: float
    t_sec: float

    def factors(self, reference: torch.Tensor) -> torch.Tensor:
        spans = torch.tensor(
            [self.x_km, self.y_km, self.z_km, self.t_sec],
            dtype=reference.dtype,
            device=reference.device,
        )
        if not bool(torch.all(torch.isfinite(spans))) or bool(torch.any(spans <= 0)):
            raise ValueError("Physical coordinate spans must be finite and positive.")
        return 2.0 / spans


def second_derivatives_4d(
    model: torch.nn.Module,
    coordinates: torch.Tensor,
    *,
    physical_scale: PhysicalCoordinateScale | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the spatial Hessian entries and temporal diagonal derivative."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("coordinates must have shape [N, 4].")
    if coordinates.shape[0] == 0 or not bool(torch.all(torch.isfinite(coordinates))):
        raise ValueError("coordinates must be non-empty and finite.")
    points = coordinates.detach().clone().requires_grad_(True)
    prediction = model(points)
    if prediction.ndim != 2 or prediction.shape != (points.shape[0], 1):
        raise ValueError("Derivative priors currently require model output shape [N, 1].")
    first = torch.autograd.grad(
        prediction,
        points,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]
    rows = []
    for axis in range(4):
        rows.append(
            torch.autograd.grad(
                first[:, axis : axis + 1],
                points,
                grad_outputs=torch.ones_like(first[:, axis : axis + 1]),
                create_graph=True,
                retain_graph=True,
            )[0]
        )

    factors = (
        physical_scale.factors(points)
        if physical_scale is not None
        else torch.ones(4, dtype=points.dtype, device=points.device)
    )

    result: dict[str, torch.Tensor] = {}
    names = ("x", "y", "z", "t")
    for i, first_name in enumerate(names):
        for j, second_name in enumerate(names):
            result[first_name + second_name] = rows[i][:, j : j + 1] * factors[i] * factors[j]
    return result


def derivative_prior_4d(
    model: torch.nn.Module,
    coordinates: torch.Tensor,
    *,
    mode: DerivativeMode,
    weight_horizontal: float = 1.0,
    weight_vertical: float = 1.0,
    weight_temporal: float = 1.0,
    huber_delta_vertical: float = 0.1,
    physical_scale: PhysicalCoordinateScale | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one accurately named derivative-prior formulation."""

    weights = (weight_horizontal, weight_vertical, weight_temporal)
    if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in weights):
        raise ValueError("Component weights must be finite and non-negative.")
    if not torch.isfinite(torch.tensor(huber_delta_vertical)) or huber_delta_vertical <= 0:
        raise ValueError("huber_delta_vertical must be finite and positive.")

    d = second_derivatives_4d(model, coordinates, physical_scale=physical_scale)
    components = {
        "xx": torch.mean(d["xx"] ** 2),
        "yy": torch.mean(d["yy"] ** 2),
        "zz_quadratic": torch.mean(d["zz"] ** 2),
        "tt": torch.mean(d["tt"] ** 2),
        "xy": torch.mean(d["xy"] ** 2),
        "xz": torch.mean(d["xz"] ** 2),
        "yz": torch.mean(d["yz"] ** 2),
    }
    components["zz_huber"] = functional.huber_loss(
        d["zz"], torch.zeros_like(d["zz"]), delta=huber_delta_vertical, reduction="mean"
    )

    if mode == "legacy_diagonal_4d":
        total = components["xx"] + components["yy"] + components["zz_quadratic"] + components["tt"]
    elif mode == "anisotropic_huber_4d":
        horizontal = components["xx"] + 2.0 * components["xy"] + components["yy"]
        total = (
            weight_horizontal * horizontal
            + weight_vertical * components["zz_huber"]
            + weight_temporal * components["tt"]
        )
        components["horizontal"] = horizontal
    elif mode == "spatial_hessian_3d":
        spatial = (
            components["xx"]
            + components["yy"]
            + components["zz_quadratic"]
            + 2.0 * components["xy"]
            + 2.0 * components["xz"]
            + 2.0 * components["yz"]
        )
        total = weight_horizontal * spatial + weight_temporal * components["tt"]
        components["spatial_hessian"] = spatial
    else:
        raise ValueError(f"Unknown derivative-prior mode: {mode}")
    components["total"] = total
    return total, components
