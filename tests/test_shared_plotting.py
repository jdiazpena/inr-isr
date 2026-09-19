"""Keep four xyt plot entry points and their mask/CSV product aligned."""
import importlib

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from PIL import Image
import torch


MODULES = (
    "synthetic_train_3d",
    "train_radar_3d_window_reference_reg_diagnostic",
    "synthetic_train_3d_window_reference_reg",
    "train_radar_3d_window_reg",
)


class ToyDataset:
    """Only the existing plotting helper's scaler/inverse interface is needed."""
    coord_scalers = {
        "x_km": {"min": 0.0, "max": 10.0},
        "y_km": {"min": 0.0, "max": 10.0},
        "t_sec": {"min": 0.0, "max": 1.0},
    }

    def denormalize_target(self, values):
        return np.asarray(values, dtype=np.float64)


def test_four_native_plot_entry_points_save_identical_numerical_grid_and_mask(tmp_path):
    frame = pd.DataFrame({
        "time_index": [0, 0, 0, 0],
        "t_sec": [0.0] * 4,
        "x_km": [0.0, 0.0, 10.0, 10.0],
        "y_km": [0.0, 10.0, 0.0, 10.0],
        "log10_Ne": [10.0, 10.5, 11.0, 11.5],
    })
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, 0.1, -0.1]]))
    csv_contents = []
    for name in MODULES:
        output = tmp_path/name
        output.mkdir()
        module = importlib.import_module(name)
        module.plot_xy_prediction_at_time(
            model=model, dataset=ToyDataset(), df=frame, time_index=0,
            out_dir=output, device=torch.device("cpu"), grid_nx=5, grid_ny=7,
            grid_padding_frac=0.05, nearest_radius_factor=0.4,
            grid_chunk_size=11, save_grid_csv=True, vmin=10.0, vmax=11.5)
        csv = output/"grid_prediction_time_index_0000.csv"
        png = output/"xy_time_index_0000.png"
        table = pd.read_csv(csv)
        assert len(table) == 35
        assert table.valid_mask.any() and (~table.valid_mask).any()
        assert list(table) == ["time_index", "unix_mid", "t_sec", "t_min",
                               "x_km", "y_km", "pred_log10_Ne",
                               "nearest_dist_km", "valid_mask"]
        with Image.open(png) as image:
            image.load()
            assert image.size[0] > 0 and image.size[1] > 0
        csv_contents.append(csv.read_bytes())
    assert all(contents == csv_contents[0] for contents in csv_contents)
