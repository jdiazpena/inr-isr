"""Strict default CUDA and explicit CPU controls, independent of local hardware."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from inr_isr.device import device_record, resolve_device
from inr_isr import workflow
from inr_isr.workflow import load_spec


def test_missing_cuda_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for policy in ("auto", "cuda", "cuda:0"):
        with pytest.raises(RuntimeError, match="CUDA is required"):
            resolve_device(policy)
    assert resolve_device("cpu") == torch.device("cpu")


def test_invalid_and_absent_gpu_index_fail(monkeypatch):
    with pytest.raises(ValueError, match="Invalid execution device"):
        resolve_device("gpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="index 2"):
        resolve_device("cuda:2")


def test_device_record_observes_model_and_batch():
    model = torch.nn.Linear(3, 1)
    batch = torch.ones(2, 3)
    record = device_record("cpu", torch.device("cpu"), model=model, batch=batch)
    assert record["model_device"] == "cpu"
    assert record["batch_device"] == "cpu"
    assert record["requested_device"] == "cpu"
    assert record["dtype"] == "torch.float32"


def test_query_device_is_strictly_validated_without_rewriting_recipe(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.touch()
    value = {
        "input": {"source": "synthetic", "coordinates": "xyt", "path": str(source),
                  "selection": {"window_start_index": 0, "window_size_records": 2}},
        "preprocessing": {"policy": "selected_window", "normalize_coords": True,
                          "normalize_values": True},
        "training_recipe": {"name": "active_xyt_synthetic", "parameters": {"cpu": True}},
        "query": {"kind": "observed_grid", "nx": 2, "ny": 2,
                  "chunk_size": 2, "device": "cpu"},
    }
    config = tmp_path / "case.json"
    config.write_text(json.dumps(value))
    spec = load_spec(config)
    assert spec.query["device"] == "cpu"
    assert spec.recipe_values["cpu"] is True
    value["query"]["device"] = "gpu"
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="query.device"):
        load_spec(config)


def test_public_training_missing_cuda_fails_before_output(monkeypatch, tmp_path: Path):
    source = tmp_path / "data.csv"
    source.touch()
    value = {
        "input": {"source": "synthetic", "coordinates": "xyt", "path": str(source),
                  "selection": {"window_start_index": 0, "window_size_records": 2}},
        "preprocessing": {"policy": "selected_window", "normalize_coords": True,
                          "normalize_values": True},
        "training_recipe": {"name": "active_xyt_synthetic", "parameters": {"num_steps": 1}},
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(value))
    spec = load_spec(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "training"
    with pytest.raises(RuntimeError, match="CUDA is required"):
        workflow.train(spec, output)
    assert not output.exists()


def test_query_cli_and_config_conflict_before_input_read(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.touch()
    value = {
        "input": {"source": "synthetic", "coordinates": "xyt", "path": str(source),
                  "selection": {"window_start_index": 0, "window_size_records": 2}},
        "preprocessing": {"policy": "selected_window", "normalize_coords": True,
                          "normalize_values": True},
        "training_recipe": {"name": "active_xyt_synthetic", "parameters": {}},
        "query": {"kind": "observations", "chunk_size": "full", "device": "cpu"},
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(value))
    spec = load_spec(path)
    checkpoint = tmp_path / "old.pt"
    checkpoint.touch()
    output = tmp_path / "query"
    with pytest.raises(ValueError, match="conflicts"):
        workflow.query(spec, checkpoint, output, device_policy="cuda")
    assert not output.exists()
