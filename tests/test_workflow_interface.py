"""Public four-route front-door validation without private radar or long runs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inr_isr.data import InputSpec, PreprocessingSpec
from inr_isr import workflow


def _xyt_config(path: Path) -> dict:
    """A minimal active synthetic recipe; the file is not read by validation."""
    return {
        "input": {
            "source": "synthetic", "coordinates": "xyt", "path": str(path),
            "selection": {"window_start_index": 0, "window_size_records": 2},
        },
        "preprocessing": {
            "policy": "selected_window", "normalize_coords": True,
            "normalize_values": True,
        },
        "training_recipe": {
            "name": "active_xyt_synthetic",
            "parameters": {"num_steps": 4, "hidden_features": 8, "cpu": True},
        },
        "query": {"kind": "observed_grid", "nx": 2, "ny": 3, "chunk_size": 2},
    }


def _load(tmp_path: Path, value: dict) -> workflow.WorkflowSpec:
    path = tmp_path / "case.json"
    path.write_text(json.dumps(value))
    return workflow.load_spec(path)


def test_absolute_source_spelling_and_native_defaults_are_preserved(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.touch()
    spec = _load(tmp_path, _xyt_config(source))
    assert spec.input.path == str(source)
    module, args = workflow._xyt_arguments(spec, tmp_path / "fresh")
    assert module.__name__ == "synthetic_train_3d"
    assert args.synthetic_csv == str(source)
    assert args.window_size_records == 2
    assert args.hidden_features == 8 and args.hidden_layers == 3
    assert args.num_steps == 4 and args.cpu is True
    assert args.output_dir == str(tmp_path / "fresh/training")


def test_relative_input_path_is_config_relative_only(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.touch()
    config = _xyt_config(source)
    config["input"]["path"] = "data.csv"
    assert _load(tmp_path, config).input.path == str(source)


@pytest.mark.parametrize("change", [
    lambda c: c.update(extra=1),
    lambda c: c["training_recipe"].update(name="native_xyzt"),
    lambda c: c["training_recipe"]["parameters"].update(h5_path="other.h5"),
    lambda c: c["training_recipe"]["parameters"].update(num_steps=True),
    lambda c: c["preprocessing"].update(normalize_coords=False),
    lambda c: c["query"].update(nx=0),
    lambda c: c["query"].update(chunk_size="full"),
])
def test_bad_or_ambiguous_front_door_inputs_fail_before_training(tmp_path: Path, change):
    source = tmp_path / "data.csv"
    source.touch()
    config = _xyt_config(source)
    change(config)
    with pytest.raises((ValueError, TypeError)):
        spec = _load(tmp_path, config)
        workflow._xyt_arguments(spec, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_full_batch_is_explicitly_only_for_observation_queries(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.touch()
    config = _xyt_config(source)
    config["query"] = {"kind": "observations", "chunk_size": "full"}
    spec = _load(tmp_path, config)
    assert spec.query["chunk_size"] == "full"
    config["query"]["kind"] = "observed_grid"
    config["query"].update(nx=2, ny=2)
    with pytest.raises(ValueError, match="chunk_size"):
        _load(tmp_path, config)


@pytest.mark.parametrize("mode", ["xyt", "xyzt"])
def test_observed_grid_is_x_fastest_with_physical_fixed_medians(monkeypatch, mode: str):
    physical = (np.array([[-2.0, 10.0, 1.0], [2.0, 20.0, 3.0]])
                if mode == "xyt" else
                np.array([[-2.0, 10.0, 100.0, 1.0],
                          [2.0, 20.0, 200.0, 3.0]]))
    monkeypatch.setattr(workflow, "prepare_input", lambda *_: SimpleNamespace(
        physical_field=SimpleNamespace(coordinates=physical)))
    spec = workflow.WorkflowSpec(
        InputSpec(source="radar", coordinates=mode, path="/unused"),
        PreprocessingSpec(policy=("selected_window" if mode == "xyt"
                                  else "training_split"),
                          split=(None if mode == "xyt" else {
                              "unit": "beam", "strategy": "random", "seed": 0,
                              "validation_group_count": 0,
                              "calibration_group_count": 1,
                              "test_group_count": 1,
                          })),
        "active_xyt_radar" if mode == "xyt" else "native_xyzt",
        {}, {"kind": "observed_grid", "nx": 2, "ny": 2, "chunk_size": 2}, {},
    )
    grid = workflow._query_coordinates(spec)
    assert grid.shape == (4, physical.shape[1])
    np.testing.assert_array_equal(grid[:, :2],
                                  [[-2, 10], [2, 10], [-2, 20], [2, 20]])
    np.testing.assert_array_equal(grid[:, 2], np.median(physical[:, 2]))
    if mode == "xyzt":
        np.testing.assert_array_equal(grid[:, 3], np.median(physical[:, 3]))


def test_occupied_and_sealed_outputs_fail_without_touching_them(tmp_path: Path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        workflow._fresh_output(occupied)
    (tmp_path / "R5_FINAL_MANIFEST.json").touch()
    with pytest.raises(FileExistsError, match="frozen"):
        workflow._fresh_output(tmp_path / "new")
    assert not (tmp_path / "new").exists()
