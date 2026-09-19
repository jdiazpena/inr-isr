"""Characterize the two old xyt analyzer APIs after mechanics are shared."""
import numpy as np
import pandas as pd
import pytest
import torch

import synthetic_analyze_reconstruction as density
import synthetic_analyze_reconstruction_log_errors as log


def test_both_analyzers_use_the_same_physical_chain_rule_without_changing_mode():
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.25, 0.75]]))
    model.train()
    points = np.arange(21, dtype=np.float32).reshape(7, 3) / 21
    axes = {name: {"min": 0.0, "max": span}
            for name, span in (("x_km", 10.0), ("y_km", 20.0), ("t_sec", 4.0))}
    target = {"min": 10.0, "max": 12.0}
    results = [module.predict_log10_and_gradients_on_coords(
        model, points, axes, target, torch.device("cpu"), 3)
        for module in (density, log)]
    assert model.training
    assert list(results[0]) == ["pred_log10_Ne", "pred_dlog10Ne_dx_km",
                                "pred_dlog10Ne_dy_km", "pred_dlog10Ne_dt_sec"]
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name])
    np.testing.assert_allclose(results[0]["pred_dlog10Ne_dx_km"], 0.1)
    np.testing.assert_allclose(results[0]["pred_dlog10Ne_dy_km"], -0.025)
    np.testing.assert_allclose(results[0]["pred_dlog10Ne_dt_sec"], 0.375)


@pytest.mark.parametrize("module", (density, log))
def test_analyzer_metric_schema_and_value_order(module):
    errors = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    actual = module.compute_metrics(errors, "axis_")
    x = errors.astype(np.float64)
    absolute = np.abs(x)
    expected = {
        "axis_mse": float(np.mean(x ** 2)),
        "axis_rmse": float(np.sqrt(np.mean(x ** 2))),
        "axis_mae": float(np.mean(absolute)),
        "axis_bias": float(np.mean(x)),
        "axis_max_abs": float(np.max(absolute)),
        "axis_p95_abs": float(np.quantile(absolute, 0.95)),
        "axis_p99_abs": float(np.quantile(absolute, 0.99)),
    }
    assert actual == expected and list(actual) == list(expected)


def test_analyzer_time_selection_retains_explicit_and_mutable_policies(monkeypatch):
    frame = pd.DataFrame({"time_index": [9, 1, 7, 3, 5]})
    assert density.select_time_indices(frame, [7, 1, 7]) == [7, 1, 7]
    assert density.select_time_indices(frame, "first_middle_last") == [1, 5, 9]
    monkeypatch.setattr(log, "TIME_SELECTION", "all")
    assert log.select_time_indices(frame) == [1, 3, 5, 7, 9]
    monkeypatch.setattr(log, "TIME_SELECTION", "invalid")
    with pytest.raises(ValueError, match="Unsupported TIME_SELECTION: invalid"):
        log.select_time_indices(frame)
    with pytest.raises(ValueError, match="Unsupported time selection: invalid"):
        density.select_time_indices(frame, "invalid")
    with pytest.raises(ValueError, match=r"Requested time_index values are unavailable: \[8\]"):
        density.select_time_indices(frame, [8])


def test_analyzer_grid_retains_float64_shape_order_and_error_words(monkeypatch):
    config = {"domain_size_km": 8.0}
    axes = {name: {"min": lower, "max": upper}
            for name, lower, upper in (("x_km", -1.0, 2.0), ("y_km", 3.0, 5.0))}
    x, y = density.make_dense_grid(config, axes, 3, 2, "training_extent")
    assert x.shape == y.shape == (2, 3)
    assert x.dtype == y.dtype == np.float64
    np.testing.assert_array_equal(x[0], [-1.0, 0.5, 2.0])
    np.testing.assert_array_equal(y[:, 0], [3.0, 5.0])
    monkeypatch.setattr(log, "GRID_NX", 3)
    monkeypatch.setattr(log, "GRID_NY", 2)
    monkeypatch.setattr(log, "DOMAIN_MODE", "training_extent")
    xx, yy = log.make_dense_grid(config, axes)
    np.testing.assert_array_equal(xx, x)
    np.testing.assert_array_equal(yy, y)
    monkeypatch.setattr(log, "DOMAIN_MODE", "invalid")
    with pytest.raises(ValueError, match="Unsupported DOMAIN_MODE: invalid"):
        log.make_dense_grid(config, axes)
    with pytest.raises(ValueError, match="Unsupported domain mode: invalid"):
        density.make_dense_grid(config, axes, domain_mode="invalid")
