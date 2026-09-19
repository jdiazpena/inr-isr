"""Named input selection/ownership/fit policies, separate from private parity audit.

Generated examples exercise deliberate edge cases. R2's private acceptance also
requires both actual radar products and all four training/reload/query workflows;
these tests do not replace those checks or claim scientific accuracy.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inr_isr.data import InputSpec, PhysicalField, PreprocessingSpec, prepare_input
from inr_isr.data.__main__ import main
from inr_isr.data.arrays import immutable_array
from inr_isr_4d.data import FieldBundle4D


def csv_fixture(tmp_path):
    """Write an unordered CSV with a deliberate invalid coordinate and extra truth."""
    path = tmp_path / "observations.csv"
    pd.DataFrame(dict(x_km=[2., 4., np.nan, 1.], y_km=[3., 2., 8., 5.],
                      t_sec=[20., 0., 0., 20.], time_index=[7, 3, 3, 7],
                      log10_Ne=[11., 12., 13., 14.], Ne_error=[np.nan, 1., 2., 3.],
                      truth=[10., 11., 12., 13.])).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("coords,targets", [(False, False), (False, True), (True, False), (True, True)])
def test_xyt_named_data_keeps_order_columns_and_normalization_switches(tmp_path, coords, targets):
    """Window fit is every valid selected row; missing error does not drop that row."""
    path = csv_fixture(tmp_path)
    prepared = prepare_input(InputSpec("synthetic", "xyt", path, dict(verbose=False)),
                             PreprocessingSpec("selected_window", coords, targets))
    assert prepared.physical_field.axes == ("x_km", "y_km", "t_sec")
    assert prepared.physical_field.units == ("km", "km", "s")
    np.testing.assert_array_equal(prepared.physical_field.row_identity["csv_row_position"], [0, 1, 3])
    assert list(prepared.native.df.columns)[-2:] == ["Ne_error", "truth"]
    assert np.isnan(prepared.native.df.Ne_error.iloc[0])
    assert prepared.coordinates is prepared.native.coords and prepared.targets is prepared.native.values
    assert prepared.coordinates.dtype == torch.float32
    np.testing.assert_array_equal(prepared.train_indices, [0, 1, 2])
    assert prepared.native.coords_raw.flags.writeable
    assert not prepared.physical_field.coordinates.flags.writeable
    assert prepared.physical_field.metadata["dataframe"] is prepared.native.df
    if not coords:
        np.testing.assert_array_equal(prepared.coordinates.numpy(), prepared.native.coords_raw.astype(np.float32))
    if not targets:
        np.testing.assert_array_equal(prepared.targets.numpy(), prepared.native.values_raw.astype(np.float32))
    trace = prepared.trace_row(1)
    assert trace["source_identity"]["csv_row_position"] == 1 and trace["row_in_fit_population"]
    with pytest.raises(IndexError):
        prepared.trace_row(-1)


def test_csv_record_position_window_clips_end_without_reordering_rows(tmp_path):
    """Sorted unique record positions select CSV rows; an oversize end is clipped."""
    path = csv_fixture(tmp_path)
    prepared = prepare_input(InputSpec("synthetic", "xyt", path,
                             dict(window_start_index=1, window_size_records=50, verbose=False)),
                             PreprocessingSpec("selected_window", False, False))
    np.testing.assert_array_equal(prepared.physical_field.row_identity["csv_row_position"], [0, 3])
    assert prepared.native.df.time_index.tolist() == [7, 7]


def test_legacy_nonfinite_retention_is_explicit_and_does_not_change_old_dataset(tmp_path):
    """drop_bad_rows=False retains NaN and readonly carrier copies it, not cleans it."""
    prepared = prepare_input(InputSpec("synthetic", "xyt", csv_fixture(tmp_path),
                             dict(drop_bad_rows=False, verbose=False)), PreprocessingSpec("selected_window", False, False))
    assert len(prepared.coordinates) == 4 and np.isnan(prepared.physical_field.coordinates[2, 0])
    assert prepared.physical_field.allow_nonfinite


@pytest.mark.parametrize("variant", ["gaussian_3d", "chapman_f2_reference"])
def test_xyzt_uses_explicit_training_only_fit_and_native_metadata(variant):
    """Both families retain native truth/integration, and held-out rows are not fit."""
    rng = torch.get_rng_state().clone()
    prepared = prepare_input(InputSpec("synthetic", "xyzt", field=dict(variant=variant),
                             observation=dict(n_beams=7, n_ranges=3, n_times=3)),
                             PreprocessingSpec("training_split", split=dict(unit="beam", strategy="random", seed=0,
                                 validation_group_count=1, calibration_group_count=1, test_group_count=1)))
    field = prepared.physical_field
    assert field.axes == ("x_km", "y_km", "z_km", "t_sec") and len(field.coordinates) == 63
    assert len(prepared.train_indices) == 36
    expected = prepared.native.bundle.coordinates[prepared.train_indices]
    np.testing.assert_array_equal(prepared.problem.coordinate_scaler.minimum, expected.min(axis=0))
    np.testing.assert_array_equal(prepared.problem.coordinate_scaler.maximum, expected.max(axis=0))
    np.testing.assert_array_equal(prepared.coordinates.numpy(),
                                 prepared.problem.coordinate_scaler.transform(field.coordinates).astype(np.float32))
    assert field.metadata["native_case"] is prepared.native
    assert not prepared.native.bundle.coordinates.flags.writeable
    np.testing.assert_array_equal(torch.get_rng_state().numpy(), rng.numpy())


@pytest.mark.parametrize("kwargs", [dict(source="radar", coordinates="xyz"),
    dict(source="unknown", coordinates="xyt"), dict(source="radar", coordinates="xyzt"),
    dict(source="synthetic", coordinates="xyzt", path="wrong", field={}, observation={}),
    dict(source="synthetic", coordinates="xyt", path="unread", selection=dict(normalize_coords=False)),
    dict(source="synthetic", coordinates="xyt", path="unread", selection=dict(coord_cols=["t_sec", "y_km", "x_km"])),
    dict(source="synthetic", coordinates="xyt", path="unread", selection=dict(target_col="Ne"))])
def test_bad_named_source_configs_fail_before_reading(kwargs):
    """No ambiguous axes/target, hidden normalization or guessed generator source."""
    with pytest.raises(ValueError):
        InputSpec(**kwargs)


def test_explicit_policy_mismatch_and_structure_fail_before_reading():
    """A source type cannot implicitly choose the fit population."""
    with pytest.raises(ValueError, match="xyt requires"):
        prepare_input(InputSpec("radar", "xyt", "unread"), PreprocessingSpec("training_split", split={}))
    for kwargs in [dict(policy="guess"), dict(policy="selected_window", split={}),
                   dict(policy="training_split"), dict(policy="training_split", split={}, normalize_values=False)]:
        with pytest.raises(ValueError):
            PreprocessingSpec(**kwargs)


def test_physical_carrier_rejects_mislabeled_shapes_and_owns_numeric_copies():
    """Owned arrays do not mutate caller values; row identities must align."""
    coords = np.arange(6.).reshape(2, 3)
    field = PhysicalField("synthetic", "xyt", coords, np.ones((2, 1)), {"row": np.arange(2)})
    coords[0, 0] = 100
    assert field.coordinates[0, 0] == 0
    with pytest.raises(ValueError):
        PhysicalField("synthetic", "xyzt", coords, np.ones((2, 1)), {"row": np.arange(2)})
    with pytest.raises(ValueError):
        PhysicalField("synthetic", "xyt", coords, np.ones((2, 2)), {"row": np.arange(2)})
    with pytest.raises(ValueError):
        PhysicalField("synthetic", "xyt", coords, np.ones((2, 1)), {"row": np.arange(3)})


def test_old4d_validator_alias_keeps_module_identity_and_original_default_errors():
    """Moved helper still rejects nonfinite values; old FieldBundle4D identity remains."""
    from inr_isr_4d.data import _immutable_array
    assert _immutable_array is immutable_array
    assert FieldBundle4D.__module__ == "inr_isr_4d.data"
    with pytest.raises(ValueError, match="probe must have 2 dimensions, got 1."):
        _immutable_array([np.nan], ndim=2, name="probe")
    with pytest.raises(ValueError, match="probe contains non-finite values."):
        _immutable_array([[np.nan]], ndim=2, name="probe")


def test_inspector_rejects_existing_or_frozen_outputs_before_reading(tmp_path):
    """A new data inspector must not write into existing reference/inspection files."""
    (tmp_path / "REFERENCE_MANIFEST.json").write_text("{}")
    with pytest.raises(FileExistsError):
        main(["--config", str(tmp_path / "missing.json"), "--output", str(tmp_path / "child")])
    assert not (tmp_path / "child").exists()


def test_row_trace_retains_python_object_string_identity():
    """Object-dtype CSV identifiers can be Python strings, not NumPy scalars."""
    from inr_isr.data import PreparedInput
    field = PhysicalField("synthetic", "xyt", np.ones((1, 3)), np.ones((1, 1)),
                          {"beam_id": np.array(["beam-a"], dtype=object)})
    prepared = PreparedInput(field, torch.ones((1, 3)), torch.ones((1, 1)), np.array([0]),
                             {"policy": "selected_window"}, None)
    assert prepared.trace_row(0)["source_identity"]["beam_id"] == "beam-a"
