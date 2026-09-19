"""Trusted local checkpoint read and legacy3D architecture reconstruction.

Pickle-enabled torch.load must only consume trusted project artifacts. This module
does not rewrite checkpoints, save training state, repair schemas or bypass4D
calibration/model identity checks; those stay with the original native consumers.
"""
from __future__ import annotations
from pathlib import Path
import torch
from .model import MLPINR, restore_parameters


def torch_load_checkpoint(path: Path, device, *, legacy_keyword_fallback=True):
    """Read trusted state at caller's old map_location, weights_only=False.

3D historically retries TypeError without weights_only;4D explicitly disables
that fallback. Preserve those error contracts instead of blanket compatibility.
No model initialization/RNG, mutation of the file or schema validation occurs.
"""
    if not legacy_keyword_fallback:
        return torch.load(path, map_location=device, weights_only=False)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """Rebuild old3D analyzer INR using config/scalar scaler/model_state_dict.

Input count remains len(coord_scalers), including historical nonstandard counts;
the new named-mode API does not reject old artifacts. Validate required scalers
before construction; then retain constructor defaults/coercions/RNG, strict state
loading, device move and eval. Target remains scalar normalized log density.
"""
    cfg = checkpoint.get("config", {})
    coord_scalers = checkpoint.get("coord_scalers")
    target_scaler = checkpoint.get("target_scaler")
    if coord_scalers is None:
        raise KeyError("Checkpoint is missing coord_scalers.")
    if target_scaler is None:
        raise KeyError("Checkpoint is missing target_scaler.")
    model = MLPINR(in_features=len(coord_scalers), out_features=1,
        hidden_features=int(cfg.get("hidden_features", 256)),
        hidden_layers=int(cfg.get("hidden_layers", 3)), activation=str(cfg.get("activation", "sine")),
        first_omega_0=float(cfg.get("first_omega_0", 5.0)),
        hidden_omega_0=float(cfg.get("hidden_omega_0", 5.0)), outermost_linear=True)
    return restore_parameters(model, checkpoint, device=device, evaluation=True, parameter_key="model_state_dict")
