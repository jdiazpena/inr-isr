"""Audited arithmetic/data checks. Never construct or invoke a neural module."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch

from inr_isr.results_xyt.common import fresh_directory, sha256
from inr_isr.results_xyt.config import validate, training_settings
from inr_isr.results_xyt.data import (exposure_points, fit_scaler, inverse_targets,
                                    normalize_targets, observation_truth, quadrature_weights,
                                    read_tables)
from inr_isr.results_xyt.evaluation import analyze, frozen_final, score_comparison
from inr_isr.results_xyt.math import (LinearBaseline, feature_summary, metrics,
                                    numpy_log_average, tensor_observation_log10,
                                    tensor_observation_mse)


@pytest.fixture(autouse=True)
def prohibit_neural_modules(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Neural construction/execution is outside this test scope")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)


@pytest.fixture
def case():
    zero = "0" * 64
    return {"schema_version": 1, "case_id": "unit-arithmetic",
            "input": dict(path="input.h5", sha256=zero, reader_sha256=zero,
                          h0_km=330., half_width_km=15., utc_start="start", utc_end="end"),
            "tables": {k: dict(path=f"{k}.csv", sha256=zero) for k in ("geometry", "times")},
            "field": dict(background_ne_m3=1e11, noise_std_log10=0,
                          patch=dict(name="clean", amplitude_m3=4e11, sigma_x_km=60., sigma_y_km=60.,
                                     x0_km=0., y0_km=0., vx_km_s=.028, vy_km_s=0.)),
            "bounds": [[-1., 1.], [-1., 1.], [-1., 11.]],
            "queries": {k: dict(count=8, seed=i, coordinate_sha256=zero)
                        for i, k in enumerate(("development", "final"))},
            "training": dict(model=dict(activation="sine", hidden_features=256, hidden_layers=3,
                                        first_omega_0=5., hidden_omega_0=5.), seed=0, lr=1e-4,
                             microbatch_observations=128, max_updates=3000, max_seconds=1200.,
                             validation_every=100, device="cuda", dtype="float32", amp=False,
                             regularization=dict(lambda_curv_xy=0., lambda_curv_t=0., reference_loss_weights=False,
                                                 target_xy_ratio=0., target_t_ratio=0., reg_ramp_frac=0.)),
            "diagnostics": dict(slice_fractions=[.15, .5, .85], grid_size=64, trace_beam_indices=[0, 1])}


@pytest.fixture
def training_rows():
    rows = []
    for beam, (x, y) in enumerate(((0., 0.), (1., 0.), (0., 1.), (1., 1.))):
        for t in (0., 5., 10.):
            rows.append(dict(beam_index=beam, x_km=x, y_km=y, t_sec=t, start_sec=t-1,
                             end_sec=t+1, log10_Ne=10+2*x+3*y+.1*t, role="train"))
    return pd.DataFrame(rows)


def test_train_only_scaler_and_no_heldout_clipping():
    roles = ["train", "validation", "train", "final_test"]
    values = np.array([10., 11., 12., 14.])
    reference = fit_scaler(values, roles)
    values[[1, 3]] = [np.nan, -1e9]
    assert fit_scaler(values, roles) == reference == {"min": 10., "max": 12., "fit_count": 2}
    np.testing.assert_array_equal(normalize_targets([10., 12., 14.], reference), [-1, 1, 3])
    np.testing.assert_array_equal(inverse_targets([-1., 1., 3.], reference), [10., 12., 14.])


@pytest.mark.parametrize("values,roles", [([], []), ([11., 11.], ["train", "train"]),
                                         ([np.nan, 12.], ["train", "train"])])
def test_scaler_rejects_invalid_fit(values, roles):
    with pytest.raises(ValueError): fit_scaler(values, roles)


def test_exposure_nodes_and_linear_density_trapezoid():
    rows = pd.DataFrame(dict(start_sec=[-63.5, 100.], end_sec=[63.5, 228.], x_km=[1., 2.], y_km=[3., 4.]))
    points = exposure_points(rows)
    np.testing.assert_array_equal(points[:, 0, 2], rows.start_sec)
    np.testing.assert_array_equal(points[:, -1, 2], rows.end_sec)
    weights = quadrature_weights()
    assert len(weights) == 33 and weights.sum() == 1 and weights[0] == weights[1]/2
    density = 10000 + 20 * points[..., 2]
    np.testing.assert_allclose(numpy_log_average(np.log10(density)), np.log10(density.mean(axis=1)), atol=1e-14)


def test_tensor_inverse_then_average_and_stability():
    scaler = dict(min=10., max=12.)
    nodes = torch.linspace(-1, 1, 66, dtype=torch.float64).reshape(2, 33).requires_grad_()
    expected = numpy_log_average(10 + (nodes.detach().numpy()+1))
    actual = tensor_observation_log10(nodes, scaler)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-13)
    assert not np.allclose(expected, (10+(nodes.detach().numpy()+1)).mean(axis=1))
    actual.sum().backward()
    assert torch.isfinite(nodes.grad).all() and (nodes.grad > 0).all()
    large = torch.full((1, 33), 1000., dtype=torch.float64, requires_grad=True)
    value = tensor_observation_log10(large, scaler)
    assert value.item() == pytest.approx(1011.)
    value.sum().backward()
    assert torch.isfinite(large.grad).all()


def test_tensor_autograd_gradcheck():
    nodes = torch.linspace(-.5, .5, 66, dtype=torch.float64).reshape(2, 33).requires_grad_()
    assert torch.autograd.gradcheck(lambda p: tensor_observation_log10(p, dict(min=10., max=12.)), (nodes,))


def test_unequal_microbatches_reproduce_full_tensor_gradient():
    nodes = torch.linspace(-1, 1, 231, dtype=torch.float64).reshape(7, 33).requires_grad_()
    targets = torch.linspace(10., 12., 7, dtype=torch.float64)
    scaler = dict(min=10., max=12.)
    full = tensor_observation_mse(nodes, targets, scaler)
    gradient = torch.autograd.grad(full, nodes)[0]
    combined = sum(tensor_observation_mse(nodes[a:b], targets[a:b], scaler)*(b-a)/7
                   for a, b in ((0, 3), (3, 6), (6, 7)))
    torch.testing.assert_close(combined, full, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(combined, nodes)[0], gradient, atol=1e-14, rtol=1e-14)


def test_baseline_affine_reproduction_and_support(training_rows):
    baseline = LinearBaseline(training_rows)
    points = np.array([[.2, .3, 4.], [2., .3, 4.], [.2, .3, -1.], [.2, .3, 11.1]])
    values, support, extension = baseline.predict(points)
    assert values[0] == pytest.approx(10+2*.2+3*.3+.1*4)
    assert support.tolist() == [True, False, False, False] and not extension.any()
    values, support, extension = baseline.predict(points, endpoint_extension=True)
    assert support.tolist() == [True, False, True, False]
    assert extension.tolist() == [False, False, True, False]
    assert values[2] == pytest.approx(10+2*.2+3*.3)


def test_baseline_matched_averaging_and_boundary_flags(training_rows):
    baseline = LinearBaseline(training_rows)
    rows = training_rows[training_rows.beam_index == 0]
    values, support, extended = baseline.observations(rows)
    assert support.all() and extended.tolist() == [True, False, True]
    expected = numpy_log_average(baseline.predict(exposure_points(rows).reshape(-1, 3), True)[0].reshape(3, 33))
    np.testing.assert_allclose(values, expected, atol=1e-14)
    assert values[1] != pytest.approx(rows.log10_Ne.iloc[1], abs=1e-5)
    assert baseline.constant == training_rows.log10_Ne.mean()


def test_baseline_rejects_validation_fit(training_rows):
    training_rows.loc[0, "role"] = "validation"
    with pytest.raises(ValueError): LinearBaseline(training_rows)


def test_macro_mean_mse_then_root_and_physical_units():
    truth = np.zeros(3)
    result, groups = metrics(truth, [0., 0., 2.], np.array([0, 0, 1]))
    assert result["log10_pooled"]["rmse"] == pytest.approx(np.sqrt(4/3))
    assert result["log10_macro"]["rmse"] == pytest.approx(np.sqrt(2))
    assert result["log10_macro"]["mae"] == 1
    assert result["linear_macro_m3"]["rmse"] == pytest.approx(99/np.sqrt(2))
    assert result["linear_m3"]["bias"] == 33
    assert [r["count"] for r in groups] == [2, 1]


def test_common_support_counts_and_no_silent_drop():
    result, _, mask = score_comparison(np.zeros(3), {"a": np.zeros(3), "b": np.array([1., np.nan, 1.])},
                                       {"a": np.ones(3, bool), "b": np.array([True, False, True])})
    assert mask.tolist() == [True, False, True]
    assert result["total_rows"] == 3 and result["excluded_rows"] == 1
    with pytest.raises(ValueError): metrics([0., np.nan], [0., 0.])


@pytest.mark.parametrize("key,value", [("device", "cpu"), ("device", "cuda:bad"), ("amp", True),
                                       ("max_updates", 0), ("microbatch_observations", True), ("dtype", "float64")])
def test_strict_training_config(case, key, value):
    case["training"][key] = value
    with pytest.raises(ValueError): validate(case)


@pytest.mark.parametrize("key", ["lambda_curv_xy", "lambda_curv_t", "reference_loss_weights", "target_xy_ratio", "target_t_ratio", "reg_ramp_frac"])
def test_config_rejects_each_fixed_or_adaptive_prior(case, key):
    case["training"]["regularization"][key] = True if key == "reference_loss_weights" else .1
    with pytest.raises(ValueError): validate(case)


def test_valid_config_and_smoke_preserves_protocol(case):
    validate(case)
    before = copy.deepcopy(case)
    settings = training_settings(case, smoke=True)
    assert settings["max_updates"] == 2 and settings["validation_every"] == 1
    assert settings["model"] == case["training"]["model"] and settings["regularization"] == case["training"]["regularization"]
    assert case == before


def test_fresh_output_does_not_overwrite(tmp_path):
    folder = fresh_directory(tmp_path/"run")
    marker = folder/"marker"; marker.write_text("preserved")
    with pytest.raises(FileExistsError): fresh_directory(folder)
    assert marker.read_text() == "preserved"


def test_final_gate_precedes_target_generation(case, tmp_path):
    final = pd.DataFrame(dict(role=["final_test"]))
    with pytest.raises(ValueError, match="Reserved"): observation_truth(case, final)
    with pytest.raises(ValueError, match="freeze"): frozen_final(None, "identity", tmp_path/"absent.pt")
    with pytest.raises(ValueError, match="freeze"):
        analyze(case, "identity", tmp_path/"absent-data", tmp_path/"absent.pt", tmp_path/"out", final=True)
    assert not (tmp_path/"out").exists()
    checkpoint = tmp_path/"checkpoint"; checkpoint.write_text("test binding only")
    freeze = tmp_path/"freeze.json"
    record = dict(comparison_frozen=True, config_identity="identity", checkpoint_sha256=sha256(checkpoint),
                  operator="explicit user", frozen_utc="2026-10-07T00:00:00Z")
    freeze.write_text(json.dumps(record))
    assert frozen_final(freeze, "identity", checkpoint) == sha256(freeze)
    with pytest.raises(ValueError): frozen_final(freeze, "different", checkpoint)


def test_feature_peak_width_and_truncation():
    axis = np.linspace(-6, 6, 241)
    x, y = np.meshgrid(axis, axis)
    image = np.log10(1e11 + 4e11*np.exp(-.5*(x*x+y*y)))
    result = feature_summary(axis, axis, image, 1e11)
    assert result["peak_x_km"] == 0 and result["peak_y_km"] == 0
    assert result["excess_amplitude_m3"] == pytest.approx(4e11)
    assert result["fwhm_x_km"] == pytest.approx(2*np.sqrt(2*np.log(2)), abs=.001)
    cropped = image.copy(); cropped[:, axis > .5] = np.nan
    assert feature_summary(axis, axis, cropped, 1e11)["width_undefined_or_truncated"]


def test_duplicate_metadata_rejected_before_input_read(case, tmp_path):
    geometry = pd.DataFrame(dict(beam_index=[0, 0], beamcode=[1, 2], spatial_role=["train", "validation"]))
    times = pd.DataFrame(dict(original_record_index=[0, 1]))
    for name, frame in (("geometry", geometry), ("times", times)):
        path = tmp_path/f"{name}.csv"; frame.to_csv(path, index=False)
        case["tables"][name] = dict(path=str(path), sha256=sha256(path))
    with pytest.raises(ValueError, match="Duplicate"): read_tables(case, verify_input=False)


def test_cuda_unavailable_fails_without_fallback(monkeypatch):
    from inr_isr.device import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"): resolve_device("cuda")
