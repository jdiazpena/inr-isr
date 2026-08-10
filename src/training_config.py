"""Typed defaults and optional JSON configuration for INR training entry points.

The neural-network and loss implementations do not depend on this module. It only
defines names and defaults that were previously repeated across synthetic and radar
scripts. Existing command lines remain the authoritative interface.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SirenConfig:
    """Architecture of the coordinate MLP used as the project's primary INR."""

    activation: str = "sine"
    hidden_features: int = 256
    hidden_layers: int = 3
    first_omega_0: float = 5.0
    hidden_omega_0: float = 5.0


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer and iteration defaults shared by synthetic and radar training."""

    lr: float = 1.0e-4
    batch_size: int = 0
    num_steps: int = 10_000
    seed: int = 0


@dataclass(frozen=True)
class RegularizationConfig:
    """Curvature-prior and adaptive reference-ratio defaults."""

    lambda_curv_xy: float = 0.0
    lambda_curv_t: float = 0.0
    target_xy_ratio: float = 0.30
    target_t_ratio: float = 0.30
    epsilon_data: float = 1.0e-6
    loss_ema_beta: float = 0.99
    curvature_ema_floor: float = 1.0e-30
    lambda_smoothing: float = 0.05
    lambda_update_every: int = 10
    lambda_warmup_steps: int = 500
    freeze_lambdas_after_step: int = 0
    lambda_curv_xy_min: float = 0.0
    lambda_curv_xy_max: float = 1.0e-6
    lambda_curv_t_min: float = 0.0
    lambda_curv_t_max: float = 1.0e-6
    num_collocation: int = 8192
    collocation_grid_nx: int = 80
    collocation_grid_ny: int = 80
    reg_ramp_frac: float = 0.2


@dataclass(frozen=True)
class DiagnosticConfig:
    """Logging and derivative-health diagnostic defaults."""

    log_every: int = 10
    summary_every: int = 250
    deriv_zero_epsilon: float = 1.0e-12
    num_diagnostic_collocation: int = 4096
    component_grad_every: int = 500


@dataclass(frozen=True)
class ReconstructionGridConfig:
    """Grid used for post-training visualization, not for measured-point fitting."""

    grid_nx: int = 250
    grid_ny: int = 250
    grid_padding_frac: float = 0.05
    grid_chunk_size: int = 65_536
    nearest_radius_factor: float = 2.5
    num_plot_times: int = 3


SIREN_DEFAULTS = SirenConfig()
OPTIMIZATION_DEFAULTS = OptimizationConfig()
REGULARIZATION_DEFAULTS = RegularizationConfig()
DIAGNOSTIC_DEFAULTS = DiagnosticConfig()
GRID_DEFAULTS = ReconstructionGridConfig()


def documented_defaults() -> dict[str, object]:
    """Return the shared flat defaults using their command-line destination names."""

    merged: dict[str, object] = {}
    for config in (
        SIREN_DEFAULTS,
        OPTIMIZATION_DEFAULTS,
        REGULARIZATION_DEFAULTS,
        DIAGNOSTIC_DEFAULTS,
        GRID_DEFAULTS,
    ):
        merged.update(asdict(config))
    return merged


def parse_args_with_optional_json(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse an existing CLI with an optional JSON file of default overrides.

    `--config path.json` is consumed before the original parser runs. Values given
    explicitly on the command line override the JSON values. The returned Namespace
    does not contain an extra `config` field, preserving `run_config.json` and
    checkpoint metadata for existing commands.
    """

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining = config_parser.parse_known_args(argv)

    if config_args.config is not None:
        with config_args.config.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Training configuration must be a JSON object.")

        valid_names = {action.dest for action in parser._actions}
        unknown = sorted(set(values) - valid_names)
        if unknown:
            raise ValueError(f"Unknown training configuration keys: {unknown}")
        parser.set_defaults(**values)

    return parser.parse_args(remaining)
