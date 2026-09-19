"""Dimension-declared access to the existing, single INR network implementation.

Definitions stay at models.MLPINR and inr_isr_4d.model.SIREN4D for old imports and
pickle identities. Construction still draws PyTorch default Linear parameters
BEFORE custom weight initialization; no wrapper adds module/state-dict prefixes.
No data reading, normalization, loss selection or dropout occurs here.
"""
from __future__ import annotations
import torch
from models import ActivationName, MLPINR, Sine, get_activation, init_linear


def build_model(coordinate_mode: str, **parameters):
    """Construct the old xyt/xyzt network with explicit coordinate mode and kwargs.

Inputs declare3or4 normalized axes; outputs keep configured[N,C] capability.
Instantiation uses the original class/defaults/RNG calls, with no device/dtype
conversion until the caller asks. Invalid mode fails before initialization.
"""
    if coordinate_mode == "xyt":
        return MLPINR(in_features=3, **parameters)
    if coordinate_mode == "xyzt":
        from inr_isr_4d.model import SIREN4D
        return SIREN4D(**parameters)
    raise ValueError("coordinate_mode must be xyt or xyzt.")


def restore_parameters(model, parameters, *, device, dtype=None, evaluation=False, parameter_key=None):
    """Apply old to(device[/dtype])→strict state load→optional eval sequence.

Mutates and returns the supplied model; parameter keys must match exactly.
Construction/RNG is owned by caller. No schema adaptation or numerical casts
beyond the caller's declared old device/precision policy are invented here.
"""
    if dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.load_state_dict(parameters if parameter_key is None else parameters[parameter_key])
    if evaluation:
        model.eval()
    return model
