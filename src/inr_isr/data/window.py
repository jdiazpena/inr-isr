"""One dataframe-to-tensor kernel for the two existing selected-window datasets.

Source-specific reading, column validation and temporal selection stay in their
native constructors. This function preserves the original operation/copy order:
clean → raw float64 arrays → per-column native normalization → float32 tensors.
It is intentionally not the training-only4D scaler policy (see R3 obligations).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch


def prepare_selected_window(dataset, dataframe: pd.DataFrame, *, drop_bad_rows: bool,
                            report_drops: bool, dropped_prefix: str, empty_message: str,
                            normalize: Callable) -> None:
    """Populate the old dataset attributes without changing its public contract.

``dataset`` supplies coord_cols/target_col and normalization switches. Every
retained row fits the selected-window scalers, not just training-role rows.
Native callbacks preserve each module's normalization helpers and monkeypatch
surface. Raw arrays/dataframe remain mutable, tensors float32 and dataset[0]
remains the whole window. Errors/printing are explicit native policies. The extra
source_row_positions records incoming dataframe indices BEFORE reset: CSV row
positions for synthetic input, retained-reader positions (not HDF5 gates) for
radar. Gate identity is separately retained in the radar dataframe.
"""
    required_cols = list(dataset.coord_cols) + [dataset.target_col]
    df = dataframe.replace([np.inf, -np.inf], np.nan)
    if drop_bad_rows:
        before = len(df)
        df = df.dropna(subset=required_cols).copy()
        after = len(df)
        if report_drops and after < before:
            print(f"{dropped_prefix}: dropped {before - after} bad rows.")
    if len(df) == 0:
        raise ValueError(empty_message)

    dataset.source_row_positions = df.index.to_numpy(copy=True)
    dataset.df = df.reset_index(drop=True)
    coords_raw = dataset.df.loc[:, dataset.coord_cols].to_numpy(dtype=np.float64)
    values_raw = dataset.df.loc[:, [dataset.target_col]].to_numpy(dtype=np.float64)
    dataset.coords_raw = coords_raw.astype(np.float64)
    dataset.values_raw = values_raw.astype(np.float64)

    # Do not vectorize: preserve each old column's arithmetic and scaler schema.
    dataset.coord_scalers = {}
    coords_out = np.zeros_like(coords_raw, dtype=np.float64)
    for i, col in enumerate(dataset.coord_cols):
        if dataset.normalize_coords:
            coords_out[:, i], vmin, vmax = normalize(coords_raw[:, i], name=col)
        else:
            coords_out[:, i] = coords_raw[:, i]
            vmin = float(np.nanmin(coords_raw[:, i]))
            vmax = float(np.nanmax(coords_raw[:, i]))
        dataset.coord_scalers[col] = {"min": vmin, "max": vmax}

    if dataset.normalize_values:
        values_out, target_min, target_max = normalize(values_raw, name=dataset.target_col)
    else:
        values_out = values_raw
        target_min = float(np.nanmin(values_raw))
        target_max = float(np.nanmax(values_raw))
    dataset.target_scaler = {"column": dataset.target_col, "min": target_min, "max": target_max}
    dataset.coords = torch.from_numpy(coords_out.astype(np.float32))
    dataset.values = torch.from_numpy(values_out.astype(np.float32))
