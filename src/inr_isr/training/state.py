"""Native checkpoint persistence and RNG/CSV mechanics in one training home.

The recipe decides WHEN and WHAT to save. Atomic full xyzt continuation is not
invented for xyt: xyt CSV append and its reduced model payload stay distinct.
"""
from __future__ import annotations
import csv,json,os,random,tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
from inr_isr_4d.constraints import constraint_identities

def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    """Append one stable-schema training-history row, creating its header once."""

    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def capture_rng_state() -> dict[str, Any]:
    """Copy Python/NumPy/Torch CPU and available CUDA RNG in the native order."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore native generators after any checkpoint ``map_location`` transfer.

    PyTorch's CPU and CUDA RNG restore APIs both require CPU ByteTensors.  A
    checkpoint loaded with ``map_location='cuda'`` moves these small saved
    states onto CUDA too, so normalize only their storage device here.  The
    bytes, generator order and CUDA-availability condition remain unchanged.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])

def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    """Save unchanged torch payload, replace exact path atomically, clean temp on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_json_save(state: dict[str, Any], path: Path) -> None:
    """Write the native finite sorted JSON/newline atomically with original cleanup."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def full_xyzt_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    best_data_loss: float,
    problem: Any,
    config: Any,
    controller: Any,
    additional_constraints: tuple[Any, ...],
) -> dict[str, Any]:
    """Build unchanged schema1 full state in native evaluation order; caller decides save timing."""
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
        "additional_constraints": constraint_identities(additional_constraints),
        "resolved_config": config.to_dict(),
        "rng": capture_rng_state(),
    }
