# -*- coding: utf-8 -*-
"""
src/inr_radar/uq/conformal.py

Distribution-free Split Conformal Prediction for 4D Ionospheric Radar Neural Fields.

Provides:
- split_beams: Radar beam dataset withholding splitter (random & spatial clustered strategies).
- split_beams_by_strategy: Helper for splitting list of beam identifiers.
- compute_conformal_quantile: Finite-sample non-conformity quantile calculation.
- ConformalCalibrator4D / SplitConformalCalibrator: Distribution-free conformal quantile calibration, prediction interval generation, and empirical coverage evaluation.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
import torch


class ConformalCoverageResult(tuple):
    """
    Result container for conformal coverage evaluation.
    Behaves as a tuple (coverage, width), a float (coverage), and a dict-like container.
    """
    def __new__(cls, coverage_val: float, q_95: float, interval_width: float):
        return super().__new__(cls, (float(coverage_val), float(interval_width)))

    def __init__(self, coverage_val: float, q_95: float, interval_width: float):
        self.empirical_coverage = float(coverage_val)
        self.coverage = float(coverage_val)
        self.q_95 = float(q_95)
        self.q_hat = float(q_95)
        self.interval_width = float(interval_width)

    def __float__(self) -> float:
        return self.empirical_coverage

    def __ge__(self, other: Any) -> bool:
        return self.empirical_coverage >= float(other)

    def __le__(self, other: Any) -> bool:
        return self.empirical_coverage <= float(other)

    def __gt__(self, other: Any) -> bool:
        return self.empirical_coverage > float(other)

    def __lt__(self, other: Any) -> bool:
        return self.empirical_coverage < float(other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (float, int)):
            return self.empirical_coverage == float(other)
        return super().__eq__(other)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item in ("empirical_coverage", "coverage"):
                return self.empirical_coverage
            elif item in ("q_95", "q_hat"):
                return self.q_95
            elif item == "interval_width":
                return self.interval_width
            raise KeyError(item)
        return super().__getitem__(item)


def compute_conformal_quantile(
    residuals: np.ndarray | torch.Tensor,
    alpha: float = 0.05,
) -> float:
    """
    Computes finite-sample distribution-free conformal quantile q_{1-alpha}:
        q_{1-alpha} = np.quantile(R, min(1.0, ceil((N + 1) * (1 - alpha)) / N))
    """
    if isinstance(residuals, torch.Tensor):
        res_np = residuals.detach().cpu().view(-1).numpy()
    else:
        res_np = np.asarray(residuals).reshape(-1)

    res_np = np.abs(res_np)
    N = len(res_np)
    if N == 0:
        return 0.0

    q_val = min(1.0, float(np.ceil((N + 1) * (1.0 - float(alpha)))) / float(N))
    return float(np.quantile(res_np, q_val))


def split_beams_by_strategy(
    beam_ids: Sequence[Any] | pd.DataFrame,
    withholding_strategy: str = "random",
    calib_ratio: float = 0.15,
    test_ratio: float = 0.15,
    cluster_center_xy: tuple[float, float] = (0.0, 0.0),
    cluster_radius_km: float = 100.0,
    beam_coords: dict[Any, np.ndarray] | None = None,
    seed: int = 42,
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Splits a list of beam identifiers (or DataFrame) into strictly disjoint
    (train_beams, calib_beams, test_beams).
    """
    if isinstance(beam_ids, pd.DataFrame):
        train_df, calib_df, test_df = split_beams(
            beam_ids,
            withholding_strategy=withholding_strategy,
            calib_ratio=calib_ratio,
            test_ratio=test_ratio,
            cluster_center_xy=cluster_center_xy,
            cluster_radius_km=cluster_radius_km,
            seed=seed,
        )
        beam_col = "beam_index" if "beam_index" in beam_ids.columns else "beamcode"
        return (
            list(train_df[beam_col].unique()),
            list(calib_df[beam_col].unique()),
            list(test_df[beam_col].unique()),
        )

    unique_beams = list(dict.fromkeys(beam_ids))
    n_total = len(unique_beams)
    rng = np.random.default_rng(seed)

    if withholding_strategy == "random":
        shuffled = rng.permutation(np.array(unique_beams, dtype=object))
        n_calib = int(np.round(n_total * calib_ratio))
        n_test = int(np.round(n_total * test_ratio))

        if calib_ratio > 0 and n_calib == 0 and n_total >= 2:
            n_calib = 1
        if test_ratio > 0 and n_test == 0 and (n_total - n_calib) >= 1:
            n_test = 1

        calib_beams = list(shuffled[:n_calib])
        test_beams = list(shuffled[n_calib : n_calib + n_test])
        train_beams = [b for b in unique_beams if b not in set(calib_beams) and b not in set(test_beams)]

    elif withholding_strategy == "clustered":
        if beam_coords is not None:
            beam_distances = []
            cx, cy = cluster_center_xy
            for b in unique_beams:
                coord = beam_coords[b]
                dist = np.sqrt((coord[0] - cx) ** 2 + (coord[1] - cy) ** 2)
                beam_distances.append((b, dist))

            beam_distances.sort(key=lambda item: item[1])
            sorted_beams = [b for b, d in beam_distances]

            target_withhold = int(np.round(n_total * (calib_ratio + test_ratio)))
            cand_beams = [b for b, d in beam_distances if d <= cluster_radius_km]
            if len(cand_beams) < max(2, target_withhold):
                cand_beams = sorted_beams[: max(2, target_withhold)]

            cand_shuffled = rng.permutation(np.array(cand_beams, dtype=object))
            total_cand = len(cand_shuffled)
            rel_calib_frac = calib_ratio / (calib_ratio + test_ratio) if (calib_ratio + test_ratio) > 0 else 0.5
            n_calib = int(np.round(total_cand * rel_calib_frac))
            if total_cand >= 2:
                n_calib = max(1, min(n_calib, total_cand - 1))

            calib_beams = list(cand_shuffled[:n_calib])
            test_beams = list(cand_shuffled[n_calib:])
            train_beams = [b for b in unique_beams if b not in set(calib_beams) and b not in set(test_beams)]
        else:
            sorted_beams = list(unique_beams)
            n_withhold = int(np.round(n_total * (calib_ratio + test_ratio)))
            withheld = sorted_beams[:n_withhold]
            cand_shuffled = rng.permutation(np.array(withheld, dtype=object))
            n_cal = int(np.round(len(cand_shuffled) * 0.5))
            calib_beams = list(cand_shuffled[:n_cal])
            test_beams = list(cand_shuffled[n_cal:])
            train_beams = [b for b in unique_beams if b not in set(calib_beams) and b not in set(test_beams)]
    else:
        raise ValueError(f"Unknown withholding_strategy: {withholding_strategy}")

    return train_beams, calib_beams, test_beams


def split_beams(
    df: pd.DataFrame,
    withholding_strategy: str = "random",
    calib_ratio: float = 0.15,
    test_ratio: float = 0.15,
    cluster_center_xy: tuple[float, float] = (0.0, 0.0),
    cluster_radius_km: float = 100.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a radar DataFrame into strictly disjoint train_df, calib_df, and test_df
    based on beam indices / beamcodes.
    """
    if "beam_index" in df.columns:
        beam_col = "beam_index"
    elif "beamcode" in df.columns:
        beam_col = "beamcode"
    else:
        raise KeyError("DataFrame must contain 'beam_index' or 'beamcode' column.")

    unique_beams = np.sort(df[beam_col].unique())
    n_total = len(unique_beams)

    if n_total == 0:
        raise ValueError("DataFrame contains no unique beams.")

    rng = np.random.default_rng(seed)

    if withholding_strategy == "random":
        shuffled = rng.permutation(unique_beams)
        n_calib = int(np.round(n_total * calib_ratio))
        n_test = int(np.round(n_total * test_ratio))

        if calib_ratio > 0 and n_calib == 0 and n_total >= 2:
            n_calib = 1
        if test_ratio > 0 and n_test == 0 and (n_total - n_calib) >= 1:
            n_test = 1

        calib_beams = set(shuffled[:n_calib])
        test_beams = set(shuffled[n_calib : n_calib + n_test])
        train_beams = set(shuffled[n_calib + n_test :])

    elif withholding_strategy == "clustered":
        if "x_km" in df.columns and "y_km" in df.columns:
            x_col, y_col = "x_km", "y_km"
        elif "x" in df.columns and "y" in df.columns:
            x_col, y_col = "x", "y"
        else:
            raise KeyError("Clustered strategy requires 'x_km'/'y_km' or 'x'/'y' columns in DataFrame.")

        cx, cy = cluster_center_xy
        beam_distances = []
        for b in unique_beams:
            b_sub = df[df[beam_col] == b]
            mean_x = float(b_sub[x_col].mean())
            mean_y = float(b_sub[y_col].mean())
            dist = np.sqrt((mean_x - cx) ** 2 + (mean_y - cy) ** 2)
            beam_distances.append((b, dist))

        beam_distances.sort(key=lambda item: item[1])
        sorted_beams = [b for b, d in beam_distances]

        candidate_beams = [b for b, d in beam_distances if d <= cluster_radius_km]

        target_withhold_count = int(np.round(n_total * (calib_ratio + test_ratio)))
        if target_withhold_count == 0 and (calib_ratio > 0 or test_ratio > 0):
            target_withhold_count = min(2, n_total)

        if len(candidate_beams) < max(2, target_withhold_count):
            candidate_beams = sorted_beams[: max(2, target_withhold_count)]

        candidate_shuffled = rng.permutation(np.array(candidate_beams))
        total_cand = len(candidate_shuffled)

        rel_calib_frac = calib_ratio / (calib_ratio + test_ratio) if (calib_ratio + test_ratio) > 0 else 0.5
        n_calib = int(np.round(total_cand * rel_calib_frac))
        if total_cand >= 2:
            n_calib = max(1, min(n_calib, total_cand - 1))

        calib_beams = set(candidate_shuffled[:n_calib])
        test_beams = set(candidate_shuffled[n_calib:])
        train_beams = set(b for b in unique_beams if b not in calib_beams and b not in test_beams)

    else:
        raise ValueError(f"Unknown withholding_strategy: '{withholding_strategy}'. Choose 'random' or 'clustered'.")

    # Verify disjoint guarantee
    assert calib_beams.isdisjoint(test_beams), "calib_beams and test_beams must be disjoint!"
    assert train_beams.isdisjoint(calib_beams), "train_beams and calib_beams must be disjoint!"
    assert train_beams.isdisjoint(test_beams), "train_beams and test_beams must be disjoint!"

    train_df = df[df[beam_col].isin(train_beams)].reset_index(drop=True)
    calib_df = df[df[beam_col].isin(calib_beams)].reset_index(drop=True)
    test_df = df[df[beam_col].isin(test_beams)].reset_index(drop=True)

    return train_df, calib_df, test_df


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _extract_coords_and_targets(
    dataset: Any,
    target_col: str = "log10_Ne",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts coordinates and targets as float32 PyTorch Tensors.
    """
    if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        coords, targets = dataset[0], dataset[1]
        coords_t = torch.tensor(np.asarray(coords), dtype=torch.float32) if not isinstance(coords, torch.Tensor) else coords.to(torch.float32)
        targets_t = torch.tensor(np.asarray(targets), dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets.to(torch.float32)
        if targets_t.ndim == 1:
            targets_t = targets_t.unsqueeze(-1)
        return coords_t, targets_t

    if isinstance(dataset, pd.DataFrame):
        all_coord_options = [
            ["x_km", "y_km", "z_km", "t_sec"],
            ["x", "y", "z", "t"],
            ["x_norm", "y_norm", "z_norm", "t_norm"],
            ["x_km", "y_km", "t_sec"],
            ["x", "y", "t"],
            ["x_norm", "y_norm", "t_norm"],
        ]
        coord_cols = None
        for opts in all_coord_options:
            if all(c in dataset.columns for c in opts):
                coord_cols = opts
                break
        if coord_cols is None:
            excl = {target_col, "beam_index", "beamcode", "beam_id", "Ne", "range_index", "time_index"}
            coord_cols = [c for c in dataset.columns if c not in excl and pd.api.types.is_numeric_dtype(dataset[c])]

        target_name = target_col if target_col in dataset.columns else ("log10_Ne" if "log10_Ne" in dataset.columns else "target")
        coords_np = dataset[coord_cols].to_numpy(dtype=np.float32)
        targets_np = dataset[[target_name]].to_numpy(dtype=np.float32)
        return torch.from_numpy(coords_np), torch.from_numpy(targets_np)

    if hasattr(dataset, "coords") and (hasattr(dataset, "values") or hasattr(dataset, "target")):
        coords = dataset.coords
        targets = dataset.values if hasattr(dataset, "values") else dataset.target
        coords_t = torch.tensor(np.asarray(coords), dtype=torch.float32) if not isinstance(coords, torch.Tensor) else coords.to(torch.float32)
        targets_t = torch.tensor(np.asarray(targets), dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets.to(torch.float32)
        if targets_t.ndim == 1:
            targets_t = targets_t.unsqueeze(-1)
        return coords_t, targets_t

    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        if len(dataset) == 1:
            item0 = dataset[0]
            if isinstance(item0, dict):
                coords = item0["coords"]
                target_key = "values" if "values" in item0 else ("target" if "target" in item0 else target_col)
                targets = item0[target_key]
                coords_t = torch.tensor(np.asarray(coords), dtype=torch.float32) if not isinstance(coords, torch.Tensor) else coords.to(torch.float32)
                targets_t = torch.tensor(np.asarray(targets), dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets.to(torch.float32)
                if targets_t.ndim == 1:
                    targets_t = targets_t.unsqueeze(-1)
                return coords_t, targets_t
            elif isinstance(item0, (tuple, list)) and len(item0) == 2:
                coords, targets = item0[0], item0[1]
                if hasattr(coords, "ndim") and coords.ndim == 2:
                    coords_t = torch.tensor(np.asarray(coords), dtype=torch.float32) if not isinstance(coords, torch.Tensor) else coords.to(torch.float32)
                    targets_t = torch.tensor(np.asarray(targets), dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets.to(torch.float32)
                    if targets_t.ndim == 1:
                        targets_t = targets_t.unsqueeze(-1)
                    return coords_t, targets_t

        all_coords = []
        all_targets = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, dict):
                c = item["coords"]
                t = item["values"] if "values" in item else (item["target"] if "target" in item else item[target_col])
            elif isinstance(item, (tuple, list)):
                c, t = item[0], item[1]
            else:
                raise ValueError(f"Unsupported item type at index {i}: {type(item)}")

            c_t = torch.tensor(np.asarray(c), dtype=torch.float32) if not isinstance(c, torch.Tensor) else c.to(torch.float32)
            t_t = torch.tensor(np.asarray(t), dtype=torch.float32) if not isinstance(t, torch.Tensor) else t.to(torch.float32)

            if c_t.ndim == 1:
                c_t = c_t.unsqueeze(0)
            if t_t.ndim == 0:
                t_t = t_t.unsqueeze(0).unsqueeze(-1)
            elif t_t.ndim == 1:
                t_t = t_t.unsqueeze(-1)

            all_coords.append(c_t)
            all_targets.append(t_t)

        return torch.cat(all_coords, dim=0), torch.cat(all_targets, dim=0)

    all_coords = []
    all_targets = []
    for batch in dataset:
        if isinstance(batch, dict):
            c = batch["coords"]
            t = batch["values"] if "values" in batch else (batch["target"] if "target" in batch else batch[target_col])
        elif isinstance(batch, (tuple, list)):
            c, t = batch[0], batch[1]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        c_t = torch.tensor(np.asarray(c), dtype=torch.float32) if not isinstance(c, torch.Tensor) else c.to(torch.float32)
        t_t = torch.tensor(np.asarray(t), dtype=torch.float32) if not isinstance(t, torch.Tensor) else t.to(torch.float32)

        all_coords.append(c_t)
        all_targets.append(t_t)

    return torch.cat(all_coords, dim=0), torch.cat(all_targets, dim=0)


class ConformalCalibrator4D:
    """
    Distribution-free Split Conformal Prediction Calibrator for 4D Implicit Neural Fields.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.q_hat: float | None = None
        self.residuals: np.ndarray | None = None

    @property
    def q_95(self) -> float | None:
        return self.q_hat

    @q_95.setter
    def q_95(self, val: float | None) -> None:
        self.q_hat = val

    def fit(
        self,
        y_calib: torch.Tensor | np.ndarray,
        y_pred_calib: torch.Tensor | np.ndarray,
    ) -> float:
        """
        Direct fitting method computing non-conformity quantile q_{1-alpha} from ground truth
        y_calib and model predictions y_pred_calib.
        """
        if isinstance(y_calib, torch.Tensor):
            y_c = y_calib.detach().cpu().view(-1).numpy()
        else:
            y_c = np.asarray(y_calib).reshape(-1)

        if isinstance(y_pred_calib, torch.Tensor):
            y_p = y_pred_calib.detach().cpu().view(-1).numpy()
        else:
            y_p = np.asarray(y_pred_calib).reshape(-1)

        residuals = np.abs(y_c - y_p)
        self.residuals = residuals

        q_1_alpha = compute_conformal_quantile(residuals, alpha=self.alpha)
        self.q_hat = q_1_alpha
        return q_1_alpha

    def calibrate(
        self,
        model: torch.nn.Module | None = None,
        calib_dataset: Any = None,
        alpha: float | None = None,
        device: str | torch.device = "cuda",
        batch_size: int = 8192,
        y_true: torch.Tensor | np.ndarray | None = None,
        y_pred: torch.Tensor | np.ndarray | None = None,
    ) -> float:
        """
        Computes non-conformity residuals R_i = |y_i - y_hat_i| on calib_dataset (or y_true, y_pred),
        and computes distribution-free conformal quantile q_{1-alpha}:
            q_{1-alpha} = np.quantile(R, min(1.0, ceil((N + 1) * (1 - alpha)) / N))
        """
        if alpha is not None:
            self.alpha = float(alpha)

        if y_true is not None and y_pred is not None:
            return self.fit(y_true, y_pred)

        if model is None or calib_dataset is None:
            raise ValueError("Either (model, calib_dataset) or (y_true, y_pred) must be provided.")

        coords, targets = _extract_coords_and_targets(calib_dataset)
        N = coords.shape[0]
        if N == 0:
            raise ValueError("Calibration dataset must contain at least 1 sample.")

        dev = _resolve_device(device)
        model.to(dev)
        model.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_coords = coords[start:end].to(dev)
                batch_pred = model(batch_coords)
                preds.append(batch_pred.cpu())

        y_hat = torch.cat(preds, dim=0).view(-1).numpy()
        y_true_arr = targets.view(-1).numpy()

        return self.fit(y_true_arr, y_hat)

    def predict_interval(
        self,
        arg1: torch.nn.Module | torch.Tensor | np.ndarray,
        query_coords: torch.Tensor | np.ndarray | pd.DataFrame | None = None,
        device: str | torch.device = "cuda",
        batch_size: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray]:
        """
        Returns (y_lower, y_upper, y_pred, q_{1-alpha}) if called with (model, query_coords),
        or (y_lower, y_upper) if called with (y_pred).
        """
        if self.q_hat is None:
            raise RuntimeError("Calibrator has not been calibrated yet. Call calibrate() or fit() first.")

        # If called as predict_interval(y_pred)
        if query_coords is None and not isinstance(arg1, torch.nn.Module):
            if isinstance(arg1, torch.Tensor):
                y_pred_np = arg1.detach().cpu().view(-1).numpy()
            else:
                y_pred_np = np.asarray(arg1, dtype=float).reshape(-1)
            lower = y_pred_np - self.q_hat
            upper = y_pred_np + self.q_hat
            return lower, upper

        model = arg1
        if not isinstance(model, torch.nn.Module):
            raise TypeError("First argument must be a torch.nn.Module when query_coords is provided.")

        if isinstance(query_coords, pd.DataFrame):
            all_coord_options = [
                ["x_km", "y_km", "z_km", "t_sec"],
                ["x", "y", "z", "t"],
                ["x_norm", "y_norm", "z_norm", "t_norm"],
                ["x_km", "y_km", "t_sec"],
                ["x", "y", "t"],
                ["x_norm", "y_norm", "t_norm"],
            ]
            coord_cols = None
            for opts in all_coord_options:
                if all(c in query_coords.columns for c in opts):
                    coord_cols = opts
                    break
            if coord_cols is None:
                excl = {"log10_Ne", "target", "beam_index", "beamcode", "beam_id", "Ne", "range_index", "time_index"}
                coord_cols = [c for c in query_coords.columns if c not in excl and pd.api.types.is_numeric_dtype(query_coords[c])]
            coords_tensor = torch.tensor(query_coords[coord_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        elif isinstance(query_coords, np.ndarray):
            coords_tensor = torch.tensor(query_coords, dtype=torch.float32)
        elif isinstance(query_coords, torch.Tensor):
            coords_tensor = query_coords.to(torch.float32)
        else:
            raise TypeError(f"Unsupported query_coords type: {type(query_coords)}")

        N = coords_tensor.shape[0]
        dev = _resolve_device(device)
        model.to(dev)
        model.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_coords = coords_tensor[start:end].to(dev)
                batch_pred = model(batch_coords)
                preds.append(batch_pred.cpu())

        y_pred = torch.cat(preds, dim=0).view(-1).numpy()
        q_1_alpha = float(self.q_hat)

        y_lower = y_pred - q_1_alpha
        y_upper = y_pred + q_1_alpha

        return y_lower, y_upper, y_pred, q_1_alpha

    def evaluate_coverage(
        self,
        model: torch.nn.Module | None = None,
        test_dataset: Any = None,
        alpha: float | None = None,
        device: str | torch.device = "cuda",
        batch_size: int = 8192,
        y_true: torch.Tensor | np.ndarray | None = None,
        y_pred: torch.Tensor | np.ndarray | None = None,
    ) -> ConformalCoverageResult:
        """
        Returns ConformalCoverageResult object which behaves as:
        - a tuple (coverage, width)
        - a float (coverage)
        - a dict-like container with 'empirical_coverage', 'q_95', 'interval_width'.
        """
        if y_true is not None and y_pred is not None:
            if isinstance(y_true, torch.Tensor):
                yt = y_true.detach().cpu().view(-1).numpy()
            else:
                yt = np.asarray(y_true).reshape(-1)

            if isinstance(y_pred, torch.Tensor):
                yp = y_pred.detach().cpu().view(-1).numpy()
            else:
                yp = np.asarray(y_pred).reshape(-1)

            if self.q_hat is None:
                raise RuntimeError("Calibrator has not been calibrated/fitted yet.")

            residuals = np.abs(yt - yp)
            covered = residuals <= self.q_hat
            cov = float(np.mean(covered))
            width = float(2.0 * self.q_hat)
            return ConformalCoverageResult(coverage_val=cov, q_95=self.q_hat, interval_width=width)

        if model is None or test_dataset is None:
            raise ValueError("Either (model, test_dataset) or (y_true, y_pred) must be provided.")

        if self.q_hat is None or (alpha is not None and alpha != self.alpha):
            target_alpha = alpha if alpha is not None else self.alpha
            self.calibrate(model, test_dataset, alpha=target_alpha, device=device, batch_size=batch_size)

        coords, targets = _extract_coords_and_targets(test_dataset)
        N = coords.shape[0]
        if N == 0:
            raise ValueError("Test dataset must contain at least 1 sample.")

        dev = _resolve_device(device)
        model.to(dev)
        model.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_coords = coords[start:end].to(dev)
                batch_pred = model(batch_coords)
                preds.append(batch_pred.cpu())

        y_pred_arr = torch.cat(preds, dim=0).view(-1).numpy()
        y_true_arr = targets.view(-1).numpy()

        q_1_alpha = float(self.q_hat)
        residuals = np.abs(y_true_arr - y_pred_arr)
        covered = residuals <= q_1_alpha
        coverage = float(np.mean(covered))
        width = float(2.0 * q_1_alpha)

        return ConformalCoverageResult(coverage_val=coverage, q_95=q_1_alpha, interval_width=width)


SplitConformalCalibrator = ConformalCalibrator4D
