"""Numeric ownership/validation shared by named fields and the old4D contracts."""
from __future__ import annotations

import numpy as np


def immutable_array(value: np.ndarray, *, ndim: int, name: str,
                    require_finite: bool = True) -> np.ndarray:
    """Copy without changing dtype, check shape/finite values, then forbid writes.

The default is the original4D helper's exact operation/error order. The new
physical carrier can explicitly retain legacy nonfinite rows; this exception
does not change FieldBundle4D or scaler validation. Read-only means this array's
write flag, not deeply immutable metadata or arbitrary external Python objects.
"""
    array = np.asarray(value).copy()
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}.")
    if require_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    array.setflags(write=False)
    return array
