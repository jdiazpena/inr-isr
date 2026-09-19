"""Atomic complete-state checkpoints and completion markers."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from inr_isr.training import state as _state
import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Keep native public persistence API; recipe timing and payload are unchanged."""
    return _state.capture_rng_state()


def restore_rng_state(state: dict[str, Any]) -> None:
    """Keep native public persistence API; recipe timing and payload are unchanged."""
    return _state.restore_rng_state(state)


def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    """Keep native public persistence API; recipe timing and payload are unchanged."""
    return _state.atomic_torch_save(state, path)


def atomic_json_save(state: dict[str, Any], path: Path) -> None:
    """Keep native public persistence API; recipe timing and payload are unchanged."""
    return _state.atomic_json_save(state, path)
