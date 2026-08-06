# -*- coding: utf-8 -*-
"""
amisr_h5_reader_4d.py

Read real multi-altitude PFISR AMISR HDF5 dataset to extract a 4D volume training dataframe:
(x_km, y_km, z_km, t_sec) -> log10_Ne.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from amisr_h5_reader_3d import (
    _as_array,
    _to_km,
    _parse_utc_to_unix,
    _select_time_indices,
    _compute_radar_xyz_from_range,
    _force_beam_range_shape,
)





def read_amisr_h5_4d_volume(
    h5_path: str | Path,
    z_min_km: float = 100.0,
    z_max_km: float = 500.0,
    time_start_utc: str | float | int | None = None,
    time_end_utc: str | float | int | None = None,
    window_start_index: int | None = None,
    window_size_records: int | None = None,
    record_stride: int = 1,
    max_records: int | None = None,
    min_ne: float = 1.0e8,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract multi-altitude 4D volume observations from an AMISR HDF5 file.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"AMISR HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ne_all = _as_array(f["FittedParams"]["Ne"])
        if "dNe" in f["FittedParams"]:
            dne_all = _as_array(f["FittedParams"]["dNe"])
        else:
            dne_all = None

        range_km = _to_km(_as_array(f["FittedParams"]["Range"]), name="Range")
        altitude_km = _to_km(_as_array(f["FittedParams"]["Altitude"]), name="Altitude")

        beamcode = _as_array(f["BeamCodes"][:, 0])
        az_deg = _as_array(f["BeamCodes"][:, 1])
        el_deg = _as_array(f["BeamCodes"][:, 2])

        n_beams = len(beamcode)
        n_ranges = ne_all.shape[-1]

        range_km = _force_beam_range_shape(range_km, n_beams=n_beams, n_ranges=n_ranges, name="Range")
        altitude_km = _force_beam_range_shape(altitude_km, n_beams=n_beams, n_ranges=n_ranges, name="Altitude")

        unix_time = _as_array(f["Time"]["UnixTime"]).astype(np.float64)

        time_indices = _select_time_indices(
            unix_time=unix_time,
            time_start_utc=time_start_utc,
            time_end_utc=time_end_utc,
            record_stride=record_stride,
            max_records=max_records,
            window_start_index=window_start_index,
            window_size_records=window_size_records,
        )

        unix_mid_all = 0.5 * (unix_time[:, 0] + unix_time[:, 1])
        t0 = float(unix_mid_all[time_indices[0]])

        x_all, y_all, z_all = _compute_radar_xyz_from_range(
            range_km=range_km,
            az_deg=az_deg,
            el_deg=el_deg,
        )

        rows = []
        for time_idx in time_indices:
            ne_t = ne_all[time_idx, :, :]
            if dne_all is not None:
                dne_t = dne_all[time_idx, :, :]
            else:
                dne_t = np.full(ne_t.shape, np.nan, dtype=np.float64)

            # Filter valid points within z range
            valid = (
                np.isfinite(ne_t)
                & (ne_t > min_ne)
                & (altitude_km >= float(z_min_km))
                & (altitude_km <= float(z_max_km))
            )

            b_idx, r_idx = np.where(valid)
            if len(b_idx) == 0:
                continue

            unix_mid = float(unix_mid_all[time_idx])
            t_sec = unix_mid - t0

            selected_ne = ne_t[b_idx, r_idx]
            selected_dne = dne_t[b_idx, r_idx]
            log10_ne = np.log10(selected_ne)

            n = len(b_idx)
            block = {
                "time_index": np.full(n, int(time_idx), dtype=int),
                "unix_mid": np.full(n, unix_mid, dtype=np.float64),
                "t_sec": np.full(n, t_sec, dtype=np.float64),
                "t_hours": np.full(n, t_sec / 3600.0, dtype=np.float64),
                "beam_index": b_idx.astype(int),
                "range_index": r_idx.astype(int),
                "beamcode": beamcode[b_idx].astype(float),
                "az_deg": az_deg[b_idx].astype(float),
                "el_deg": el_deg[b_idx].astype(float),
                "range_km": range_km[b_idx, r_idx],
                "altitude_km": altitude_km[b_idx, r_idx],
                "x_km": x_all[b_idx, r_idx],
                "y_km": y_all[b_idx, r_idx],
                "z_km": z_all[b_idx, r_idx],
                "Ne": selected_ne.astype(np.float64),
                "dNe": selected_dne.astype(np.float64),
                "log10_Ne": log10_ne.astype(np.float64),
            }
            rows.append(pd.DataFrame(block))

    if len(rows) == 0:
        raise ValueError("No valid 4D AMISR volume samples extracted within requested altitude range.")

    df = pd.concat(rows, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["x_km", "y_km", "z_km", "t_sec", "log10_Ne"]).copy()
    df = df.sort_values(["time_index", "beam_index", "range_index"]).reset_index(drop=True)

    if verbose:
        print(f"Extracted 4D PFISR volume dataframe: {len(df)} rows across {df['time_index'].nunique()} times.")

    return df


class PFISRVolume4DDataset(Dataset):
    """
    PyTorch Dataset for real multi-altitude PFISR 4D volume training (x_km, y_km, z_km, t_sec) -> log10_Ne.
    """

    def __init__(
        self,
        h5_path: str | Path,
        z_min_km: float = 100.0,
        z_max_km: float = 500.0,
        time_start_utc: str | float | int | None = None,
        time_end_utc: str | float | int | None = None,
        window_start_index: int | None = None,
        window_size_records: int | None = None,
        record_stride: int = 1,
        max_records: int | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.df = read_amisr_h5_4d_volume(
            h5_path=h5_path,
            z_min_km=z_min_km,
            z_max_km=z_max_km,
            time_start_utc=time_start_utc,
            time_end_utc=time_end_utc,
            window_start_index=window_start_index,
            window_size_records=window_size_records,
            record_stride=record_stride,
            max_records=max_records,
            verbose=verbose,
        )

        from inr_radar.datasets.coordinate_transforms import Normalizer4D
        self.normalizer = Normalizer4D().fit(
            self.df["x_km"].to_numpy(),
            self.df["y_km"].to_numpy(),
            self.df["z_km"].to_numpy(),
            self.df["t_sec"].to_numpy(),
        )

        coords_phys = self.df[["x_km", "y_km", "z_km", "t_sec"]].to_numpy(dtype=np.float32)
        coords_norm = self.normalizer.normalize(coords_phys)

        self.val_min = float(self.df["log10_Ne"].min())
        self.val_max = float(self.df["log10_Ne"].max())
        val_span = max(self.val_max - self.val_min, 1e-6)

        values_norm = 2.0 * (self.df["log10_Ne"].to_numpy(dtype=np.float32) - self.val_min) / val_span - 1.0

        self.coords = torch.from_numpy(coords_norm.astype(np.float32))
        self.values = torch.from_numpy(values_norm.reshape(-1, 1).astype(np.float32))

        self.coord_scalers = {
            "x_km": {"min": self.df["x_km"].min(), "max": self.df["x_km"].max()},
            "y_km": {"min": self.df["y_km"].min(), "max": self.df["y_km"].max()},
            "z_km": {"min": self.df["z_km"].min(), "max": self.df["z_km"].max()},
            "t_sec": {"min": self.df["t_sec"].min(), "max": self.df["t_sec"].max()},
        }
        self.target_scaler = {"min": self.val_min, "max": self.val_max}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx == 0:
            return {"coords": self.coords, "values": self.values}
        return {"coords": self.coords[idx : idx + 1], "values": self.values[idx : idx + 1]}

    def denormalize_target(self, values_norm: np.ndarray) -> np.ndarray:
        return 0.5 * (values_norm + 1.0) * (self.val_max - self.val_min) + self.val_min

    def summary(self) -> None:
        print("PFISRVolume4DDataset")
        print(f"  rows:         {len(self.df)}")
        print(f"  coords shape: {tuple(self.coords.shape)}")
        print(f"  values shape: {tuple(self.values.shape)}")


read_amisr_hdf5_4d = read_amisr_h5_4d_volume

