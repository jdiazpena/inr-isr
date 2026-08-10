"""One canonical 4D trainer for synthetic and real observation adapters."""

from __future__ import annotations

import json
import os
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import atomic_json_save, atomic_torch_save, capture_rng_state, restore_rng_state
from .collocation import CollocationPool4D, make_collocation_pool
from .config import FourDConfig
from .controller import ReferenceRatioController
from .data import AffineScaler, FieldBundle4D
from .model import SIREN4D
from .regularization import PhysicalCoordinateScale, derivative_prior_4d, second_derivatives_4d


@dataclass(frozen=True)
class TrainingProblem4D:
    bundle: FieldBundle4D
    train_indices: np.ndarray
    coordinate_scaler: AffineScaler
    target_scaler: AffineScaler
    split_state: dict[str, Any]

    def __post_init__(self) -> None:
        indices = np.asarray(self.train_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("train_indices must be a non-empty one-dimensional array.")
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= self.bundle.size):
            raise ValueError("train_indices must be unique and within the bundle.")
        object.__setattr__(self, "train_indices", indices)


@dataclass(frozen=True)
class TrainingResult4D:
    model: SIREN4D
    output_directory: Path
    steps_completed: int
    complete: bool
    best_data_loss: float


def resolve_device(policy: str) -> torch.device:
    if policy == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(policy)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {policy} was requested but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {device.index} is unavailable.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_indices(count: int, batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if batch_size >= count:
        return torch.arange(count)
    return torch.randperm(count, generator=generator)[:batch_size]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pool_for_step(
    config: FourDConfig, train_coordinates: torch.Tensor, step: int
) -> CollocationPool4D:
    interval = config.collocation.resample_every
    generation = 0 if interval == 0 else step // interval
    return make_collocation_pool(
        mode=config.collocation.mode,
        pool_size=config.collocation.pool_size,
        seed=config.collocation.seed + generation,
        data_coordinates=train_coordinates,
        support_oversample_factor=config.collocation.support_oversample_factor,
    )


def _physical_scale(problem: TrainingProblem4D) -> PhysicalCoordinateScale:
    spans = problem.coordinate_scaler.maximum - problem.coordinate_scaler.minimum
    return PhysicalCoordinateScale(*[float(value) for value in spans])


def _derivative_health(
    model: torch.nn.Module,
    probe: torch.Tensor,
    physical_scale: PhysicalCoordinateScale | None,
    near_zero: float = 1.0e-12,
) -> dict[str, dict[str, float | bool]]:
    derivatives = second_derivatives_4d(model, probe, physical_scale=physical_scale)
    health: dict[str, dict[str, float | bool]] = {}
    for name in ("xx", "yy", "zz", "tt", "xy", "xz", "yz"):
        values = derivatives[name].detach()
        abs_values = values.abs()
        rms = float(torch.sqrt(torch.mean(values**2)).cpu())
        exact_zero = float(torch.mean((values == 0).float()).cpu())
        near_zero_fraction = float(torch.mean((abs_values < near_zero).float()).cpu())
        health[name] = {
            "rms": rms,
            "maximum_absolute": float(abs_values.max().cpu()),
            "mean_absolute": float(abs_values.mean().cpu()),
            "exact_zero_fraction": exact_zero,
            "near_zero_fraction": near_zero_fraction,
            "collapse_warning": bool(near_zero_fraction > 0.99),
        }
    return health


def _checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    best_data_loss: float,
    problem: TrainingProblem4D,
    config: FourDConfig,
    controller: ReferenceRatioController,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "next_step": next_step,
        "best_metrics": {"data_loss": best_data_loss},
        "preprocessing": {
            "coordinates": problem.coordinate_scaler.state_dict(),
            "targets": problem.target_scaler.state_dict(),
        },
        "splits": problem.split_state,
        "adaptive_controller": controller.state_dict(),
        "resolved_config": config.to_dict(),
        "rng": capture_rng_state(),
    }


def train_4d(
    problem: TrainingProblem4D,
    config: FourDConfig,
    output_directory: Path,
    *,
    resume: bool = False,
    restart: bool = False,
    stop_after_steps: int | None = None,
) -> TrainingResult4D:
    """Train or genuinely resume one 4D model.

    ``stop_after_steps`` exists for bounded interruption tests and smoke orchestration;
    it is not part of the scientific configuration.
    """

    config.validate()
    output_directory = Path(output_directory)
    checkpoint_path = output_directory / "checkpoint.pt"
    completion_path = output_directory / "COMPLETED.json"
    existing = output_directory.exists() and any(output_directory.iterdir())
    if resume and restart:
        raise ValueError("resume and restart are mutually exclusive.")
    if existing and not resume:
        raise FileExistsError(
            f"Run directory is not empty: {output_directory}. Use a new directory; historical runs are never overwritten."
        )
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {checkpoint_path}.")
    if resume and completion_path.exists():
        raise RuntimeError("A completed run cannot be resumed in place.")
    if restart and existing:
        raise FileExistsError("Explicit restart still requires a new output directory to preserve history.")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.runtime.device)
    if config.runtime.amp and device.type != "cuda":
        raise RuntimeError("AMP was requested but the resolved device is not CUDA.")
    dtype = torch.float32 if config.runtime.precision == "float32" else torch.float64
    _seed_everything(config.optimization.seed)

    model = SIREN4D(
        hidden_features=config.model.hidden_features,
        hidden_layers=config.model.hidden_layers,
        activation=config.model.activation,
        first_omega_0=config.model.first_omega_0,
        hidden_omega_0=config.model.hidden_omega_0,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimization.learning_rate)
    controller = ReferenceRatioController(config.derivative_prior)

    train_bundle = problem.bundle.subset(problem.train_indices)
    train_coordinates = torch.as_tensor(
        problem.coordinate_scaler.transform(train_bundle.coordinates), dtype=dtype
    )
    train_targets = torch.as_tensor(problem.target_scaler.transform(train_bundle.targets), dtype=dtype)

    start_step = 0
    best_data_loss = float("inf")
    if resume:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema.")
        if state["resolved_config"] != config.to_dict():
            raise ValueError("Resolved configuration differs from the checkpoint.")
        if state["splits"] != problem.split_state:
            raise ValueError("Split state differs from the checkpoint.")
        if state["preprocessing"] != {
            "coordinates": problem.coordinate_scaler.state_dict(),
            "targets": problem.target_scaler.state_dict(),
        }:
            raise ValueError("Preprocessing state differs from the checkpoint.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        controller.load_state_dict(state["adaptive_controller"])
        start_step = int(state["next_step"])
        best_data_loss = float(state["best_metrics"]["data_loss"])
        restore_rng_state(state["rng"])

    atomic_json_save(config.to_dict(), output_directory / "resolved_config.json")
    atomic_json_save(
        {
            "coordinates": problem.coordinate_scaler.state_dict(),
            "targets": problem.target_scaler.state_dict(),
            "coordinate_order": ["x_km", "y_km", "z_km", "t_sec"],
            "fit_unit": "training observations only",
        },
        output_directory / "preprocessing.json",
    )
    atomic_json_save(problem.split_state, output_directory / "splits.json")

    end_step = config.optimization.num_steps
    if stop_after_steps is not None:
        if stop_after_steps <= 0:
            raise ValueError("stop_after_steps must be positive when supplied.")
        end_step = min(end_step, start_step + stop_after_steps)

    started = time.monotonic()
    last_pool_generation: int | None = None
    pool: CollocationPool4D | None = None
    probe: torch.Tensor | None = None
    use_amp = config.runtime.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    physical = _physical_scale(problem) if config.derivative_prior.coordinate_convention == "physical" else None

    for step in range(start_step, end_step):
        generation = 0 if config.collocation.resample_every == 0 else step // config.collocation.resample_every
        if pool is None or generation != last_pool_generation:
            pool = _pool_for_step(config, train_coordinates, step)
            probe_count = min(config.runtime.diagnostic_probe_size, len(pool))
            probe = pool.sample(probe_count, step=1_000_003).to(device=device, dtype=dtype)
            last_pool_generation = generation

        indices = _batch_indices(
            len(train_coordinates),
            config.optimization.data_batch_size,
            config.optimization.seed + 10_000_000 + step,
        )
        coordinates_batch = train_coordinates[indices].to(device)
        targets_batch = train_targets[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            predictions = model(coordinates_batch)
            data_loss = torch.mean((predictions - targets_batch) ** 2)
        scaler.scale(data_loss).backward()

        raw_components = {"horizontal": 0.0, "vertical": 0.0, "temporal": 0.0}
        prior_value = 0.0
        active_weights = dict(controller.weights)
        if config.derivative_prior.mode != "none" and config.derivative_prior.weight > 0:
            assert pool is not None
            collocation = pool.sample(config.collocation.batch_size, step=step).to(device=device, dtype=dtype)
            total_count = len(collocation)
            micro = config.collocation.derivative_microbatch_size
            for begin in range(0, total_count, micro):
                points = collocation[begin : begin + micro]
                fraction = len(points) / total_count
                prior, details = derivative_prior_4d(
                    model,
                    points,
                    mode=config.derivative_prior.mode,
                    weight_horizontal=active_weights["horizontal"],
                    weight_vertical=active_weights["vertical"],
                    weight_temporal=active_weights["temporal"],
                    huber_delta_vertical=config.derivative_prior.huber_delta_vertical,
                    physical_scale=physical,
                )
                weighted = config.derivative_prior.weight * fraction * prior
                scaler.scale(weighted).backward()
                prior_value += fraction * float(prior.detach().cpu())
                raw_components["horizontal"] += fraction * float(details["horizontal"].detach().cpu())
                raw_components["vertical"] += fraction * float(details["vertical"].detach().cpu())
                raw_components["temporal"] += fraction * float(details["tt"].detach().cpu())

        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).cpu())
        scaler.step(optimizer)
        scaler.update()
        data_value = float(data_loss.detach().cpu())
        best_data_loss = min(best_data_loss, data_value)
        active_weights = controller.update(step, data_value, raw_components)

        next_step = step + 1
        should_record = next_step % config.runtime.history_every == 0 or next_step == end_step
        if should_record:
            assert probe is not None
            health = (
                _derivative_health(model, probe, physical)
                if config.derivative_prior.mode != "none"
                else {}
            )
            memory: dict[str, float | int | None] = {
                "cpu_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "gpu_allocated_bytes": None,
                "gpu_peak_bytes": None,
            }
            if device.type == "cuda":
                memory["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
                memory["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
            _append_jsonl(
                output_directory / "history.jsonl",
                {
                    "step": next_step,
                    "data_loss": data_value,
                    "derivative_prior_loss": prior_value,
                    "raw_components": raw_components,
                    "active_component_weights": active_weights,
                    "gradient_norm": gradient_norm,
                    "derivative_health": health,
                    "elapsed_seconds": time.monotonic() - started,
                    "memory": memory,
                },
            )

        if next_step % config.runtime.checkpoint_every == 0 or next_step == end_step:
            atomic_torch_save(
                _checkpoint_state(
                    model=model,
                    optimizer=optimizer,
                    next_step=next_step,
                    best_data_loss=best_data_loss,
                    problem=problem,
                    config=config,
                    controller=controller,
                ),
                checkpoint_path,
            )

    complete = end_step >= config.optimization.num_steps
    if complete:
        atomic_json_save(
            {
                "schema_version": 1,
                "status": "complete",
                "steps_completed": end_step,
                "checkpoint": checkpoint_path.name,
                "best_data_loss": best_data_loss,
            },
            completion_path,
        )
    return TrainingResult4D(model, output_directory, end_step, complete, best_data_loss)
