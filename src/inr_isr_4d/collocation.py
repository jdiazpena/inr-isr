"""Deterministic 4D collocation pools independent of observation minibatches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollocationPool4D:
    coordinates: torch.Tensor
    mode: str
    seed: int

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 4:
            raise ValueError("Collocation coordinates must have shape [N, 4].")
        if self.coordinates.shape[0] == 0 or not bool(torch.isfinite(self.coordinates).all()):
            raise ValueError("Collocation coordinates must be non-empty and finite.")

    def sample(self, count: int, *, step: int = 0) -> torch.Tensor:
        if count <= 0 or count > len(self.coordinates):
            raise ValueError("Requested collocation batch count is invalid.")
        generator = torch.Generator(device="cpu").manual_seed(self.seed + step)
        indices = torch.randperm(len(self.coordinates), generator=generator)[:count]
        return self.coordinates[indices]

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])


def _sobol(count: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    return 2.0 * engine.draw(count, dtype=dtype) - 1.0


def make_collocation_pool(
    *,
    mode: str,
    pool_size: int,
    seed: int,
    data_coordinates: torch.Tensor | None = None,
    support_oversample_factor: int = 4,
    domain_lower: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
    domain_upper: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> CollocationPool4D:
    """Create an exact-size normalized-coordinate pool using one declared mode."""

    if pool_size <= 0 or seed < 0:
        raise ValueError("pool_size must be positive and seed non-negative.")
    lower = torch.as_tensor(domain_lower, dtype=torch.float64)
    upper = torch.as_tensor(domain_upper, dtype=torch.float64)
    if lower.shape != (4,) or upper.shape != (4,) or not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
        raise ValueError("Collocation domain bounds must be finite four-vectors.")
    if bool(torch.any(lower < -1.0)) or bool(torch.any(upper > 1.0)) or bool(torch.any(lower >= upper)):
        raise ValueError("Normalized collocation bounds must satisfy -1 <= lower < upper <= 1.")

    def bounded_sobol(count: int, dtype: torch.dtype) -> torch.Tensor:
        values = _sobol(count, seed, dtype)
        low = lower.to(dtype=dtype)
        high = upper.to(dtype=dtype)
        return low + 0.5 * (values + 1.0) * (high - low)

    if mode == "data_coordinates":
        if data_coordinates is None or data_coordinates.ndim != 2 or data_coordinates.shape[1] != 4:
            raise ValueError("data_coordinates mode requires a tensor with shape [N, 4].")
        if len(data_coordinates) == 0 or not bool(torch.isfinite(data_coordinates).all()):
            raise ValueError("data_coordinates must be non-empty and finite.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if pool_size <= len(data_coordinates):
            indices = torch.randperm(len(data_coordinates), generator=generator)[:pool_size]
        else:
            indices = torch.randint(len(data_coordinates), (pool_size,), generator=generator)
        coordinates = data_coordinates.detach().cpu()[indices].clone()
    elif mode == "sobol":
        dtype = data_coordinates.dtype if data_coordinates is not None else torch.float32
        coordinates = bounded_sobol(pool_size, dtype)
    elif mode == "support_aware":
        if data_coordinates is None or data_coordinates.ndim != 2 or data_coordinates.shape[1] != 4:
            raise ValueError("support_aware mode requires observed coordinates with shape [N, 4].")
        if support_oversample_factor < 1:
            raise ValueError("support_oversample_factor must be at least one.")
        candidates = bounded_sobol(pool_size * support_oversample_factor, data_coordinates.dtype)
        observed = data_coordinates.detach().cpu()
        # Spatial support is distance to observed radar geometry; time is sampled
        # across the declared normalized domain independently.
        distances = torch.cdist(candidates[:, :3], observed[:, :3]).amin(dim=1)
        indices = torch.argsort(distances, stable=True)[:pool_size]
        coordinates = candidates[indices]
    else:
        raise ValueError(f"Unknown collocation mode: {mode}")
    return CollocationPool4D(coordinates=coordinates, mode=mode, seed=seed)
