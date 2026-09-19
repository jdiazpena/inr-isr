"""Execution-device policy shared by xyt and xyzt neural-field workflows.

An ordinary run requires CUDA so missing GPU access is visible immediately.
CPU remains an explicit choice. This module does not move data or alter RNG.
"""
from __future__ import annotations

import torch


def resolve_device(policy: str = "cuda") -> torch.device:
    """Resolve `auto`/default to required CUDA, or an explicit CPU/CUDA device.

    `auto` is retained for older configuration syntax but no longer means silent
    fallback to CPU. Call before expensive data preparation or output creation.
    """
    if policy == "auto":
        policy = "cuda"
    if policy != "cpu" and policy != "cuda" and not (
        policy.startswith("cuda:") and policy[5:].isdigit()
    ):
        raise ValueError(f"Invalid execution device: {policy!r}")
    device = torch.device(policy)
    if device.type == "cpu":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by default but is unavailable. Select CPU explicitly to use the CPU alternative.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {device.index} is unavailable.")
    return device


def device_record(policy: str, device: torch.device, *, model=None, batch=None,
                  dtype=None, amp: bool = False) -> dict[str, object]:
    """Report both selected policy and observed tensor placement, if supplied."""
    parameter = next(model.parameters(), None) if model is not None else None
    return {
        "requested_device": policy,
        "resolved_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_device": str(parameter.device) if parameter is not None else None,
        "batch_device": str(batch.device) if batch is not None else None,
        "dtype": str(dtype if dtype is not None else (parameter.dtype if parameter is not None else None)),
        "amp": bool(amp),
    }
