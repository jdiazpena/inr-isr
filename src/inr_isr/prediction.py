"""One ordered chunk/forward/collect kernel for existing deterministic INR queries.

Input is already normalized[N,D]; this kernel does not choose source/axes/fitting,
inverse scale, select eval mode or flatten scalar outputs. Native adapters retain
their mode, conversion and invalid/empty-input policies. No dropout is added.
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def predict_normalized(model, coordinates, *, device, chunk_size: int, dtype=None):
    """Return normalized[N,C] predictions in original order with exact old chunks.

dtype=None preserves3D from_numpy→to(device); declared dtype preserves4D
as_tensor(dtype/device). Callers own positivity validation and model.eval side
effects. Empty/negative/zero batches retain original NumPy/range errors at their
native boundary. No clipping, padding, reshuffling, mutation or RNG is introduced.
"""
    outputs = []
    n = coordinates.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Preserve the recipe's conversion instead of silently standardizing dtype.
        if dtype is None:
            batch = torch.from_numpy(coordinates[start:end]).to(device)
        else:
            batch = torch.as_tensor(coordinates[start:end], dtype=dtype, device=device)
        outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_log10_on_coords(model: torch.nn.Module, coords_norm: np.ndarray,
                           target_scaler: dict, device: torch.device, chunk_size: int):
    """Legacy analyzer adapter: normalizedfloat32[N,3]→float64[N] logdensity.

Does NOT call eval: old analyzer load already did, and direct callers retain
their model mode. Shared kernel disables autograd; inverse uses saved scalar
min/max state. No file access, mutation, fitting, clipping or random sampling.
"""
    from .transforms import denormalize_target
    normalized = predict_normalized(model, coords_norm, device=device, chunk_size=chunk_size)[:, 0]
    return denormalize_target(normalized, target_scaler=target_scaler)
