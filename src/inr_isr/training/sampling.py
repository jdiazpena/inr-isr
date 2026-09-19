"""Index and microbatch mechanics, independent of axes and source.

Native recipes own count validation and whether RNG is global/on-device or an
explicit CPU generator. This module does not decide which scientific pool to use.
"""
from __future__ import annotations
from collections.abc import Iterator
import torch


def without_replacement_indices(count: int, batch_size: int, *,
                                device: torch.device | None = None,
                                generator: torch.Generator | None = None,
                                full_when_nonpositive: bool = True,
                                full_when_equal: bool = True) -> torch.Tensor:
    """Compute ordered indices with the caller's original full-batch contract.

Return [K] int64 indices. xyt full/equal/nonpositive uses arange; xyzt data
retains nonpositive slicing and constructs its local generator before this call;
xyzt collocation validates first and permutes even an equal-size request.
An explicit device retains xyt's global device RNG. An explicit generator retains
xyzt's local RNG without consuming global Torch state. No data is copied here.
"""
    options={} if device is None else {'device':device}
    full=(full_when_nonpositive and batch_size<=0) or batch_size>count or (full_when_equal and batch_size==count)
    if full:
        return torch.arange(count,**options)
    if generator is not None:
        options['generator']=generator
    return torch.randperm(count,**options)[:batch_size]


def microbatch_starts(total_count: int, microbatch_size: int) -> Iterator[int]:
    """Yield original increasing slice starts, retaining range's invalid-size error.

The recipe still computes len(points)/total_count after slicing, so a short final
microbatch has its proper population fraction. No padding, shuffling, averaging
or reordering of component backward calls is introduced.
"""
    yield from range(0,total_count,microbatch_size)
