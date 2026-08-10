"""Manifest expansion, dry-run purity, orchestration, and completion tests."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inr_isr_4d.runner import case_identifier, load_manifest, main


def manifest_dict(output_root: Path, *, interrupt: bool = False) -> dict:
    case = {
        "name": "tiny synthetic",
        "question": "Does the bounded execution path produce every required artifact?",
        "data": {
            "type": "synthetic",
            "field": {"variant": "gaussian_3d"},
            "observation": {
                "mode": "instantaneous",
                "integration_duration_sec": 0.0,
                "integration_samples": 1,
                "n_beams": 5,
                "n_ranges": 3,
                "n_times": 3,
                "duration_sec": 300.0,
                "seed": 0,
            },
        },
        "split": {
            "unit": "beam",
            "strategy": "random",
            "validation_group_count": 0,
            "calibration_group_count": 1,
            "test_group_count": 1,
            "seed": 0,
        },
        "training": {
            "model": {"hidden_features": 8, "hidden_layers": 1},
            "optimization": {
                "learning_rate": 0.001,
                "num_steps": 3,
                "data_batch_size": 10,
                "seed": 0,
            },
            "collocation": {
                "mode": "sobol",
                "pool_size": 8,
                "batch_size": 4,
                "derivative_microbatch_size": 2,
                "seed": 0,
            },
            "derivative_prior": {"mode": "none", "weight": 0.0},
            "runtime": {
                "device": "cpu",
                "inference_chunk_size": 7,
                "diagnostic_probe_size": 4,
                "history_every": 1,
                "checkpoint_every": 1,
            },
        },
        "conformal": {"alpha": 0.1, "bootstrap_repetitions": 5},
    }
    if interrupt:
        case["verification"] = {"interrupt_after_steps": 1, "resume_immediately": True}
    return {
        "schema_version": 1,
        "purpose": "smoke",
        "output_root": str(output_root),
        "cases": [case],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_case_identifier_is_semantic_and_collision_safe(tmp_path: Path) -> None:
    manifest = manifest_dict(tmp_path / "outputs")
    case = manifest["cases"][0]
    first = case_identifier(case)
    assert first.startswith("tiny-synthetic-")
    changed = deepcopy(case)
    changed["training"]["optimization"]["seed"] = 1
    assert case_identifier(changed) != first


def test_dry_run_creates_no_output_artifact(tmp_path: Path, capsys) -> None:
    output = tmp_path / "must-not-exist"
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest_dict(output))
    assert main([str(path), "all", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan[0]["split_counts"]["groups"]["calibration"] == 1
    assert plan[0]["collocation"]["pool_size"] == 8
    assert not output.exists()


def test_all_stages_and_interrupted_resume_case_complete(tmp_path: Path, capsys) -> None:
    output = tmp_path / "outputs"
    path = tmp_path / "manifest.json"
    manifest = manifest_dict(output, interrupt=True)
    write_manifest(path, manifest)
    assert main([str(path), "all"]) == 0
    capsys.readouterr()
    case_directory = output / case_identifier(manifest["cases"][0])
    assert (case_directory / "GENERATED.json").is_file()
    assert (case_directory / "train" / "COMPLETED.json").is_file()
    assert (case_directory / "uq" / "calibration.json").is_file()
    assert (case_directory / "evaluation" / "predictions.npz").is_file()
    truth_predictions = np.load(case_directory / "synthetic_truth" / "predictions.npz")
    assert "instantaneous_midpoint_truth_log10_ne" in truth_predictions.files
    assert (case_directory / "SUMMARY.json").is_file()
    history = (case_directory / "train" / "history.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in history] == [1, 2, 3]
    assert main([str(path), "all"]) == 0
    rerun = json.loads(capsys.readouterr().out)
    assert all(item["status"] in {"skipped_complete", "complete"} for item in rerun)


def test_manifest_rejects_unknown_controls(tmp_path: Path) -> None:
    manifest = manifest_dict(tmp_path / "outputs")
    manifest["cases"][0]["training"]["collocation"]["unused_knob"] = 5
    path = tmp_path / "invalid.json"
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="Unknown fields"):
        load_manifest(path)


def test_restart_requires_new_explicit_attempt(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest_dict(tmp_path / "outputs"))
    with pytest.raises(ValueError, match="attempt-id"):
        main([str(path), "all", "--restart"])


def test_explicit_cli_seed_zero_and_strict_set_override(tmp_path: Path, capsys) -> None:
    manifest = manifest_dict(tmp_path / "outputs")
    manifest["cases"][0]["training"]["optimization"]["seed"] = 9
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert main(
        [
            str(path),
            "all",
            "--dry-run",
            "--seed",
            "0",
            "--set",
            "runtime.inference_chunk_size=5",
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan[0]["optimization_seed"] == 0
    assert plan[0]["inference_chunk_size"] == 5
    assert plan[0]["collocation"]["normalized_domain_lower"] == [-1.0] * 4
    assert plan[0]["collocation"]["normalized_domain_upper"] == [1.0] * 4
