"""Objective/analysis guards using scalar metadata and mocks; no neural work."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from inr_isr.experiments.contract import Contract
from inr_isr.experiments.analysis import analyze


def contract(domain=None, reference=None):
    training = {"target_col": "log10_Ne", "num_steps": 5000}
    if domain is not None:
        training["data_loss_domain"] = domain
    if reference is not None:
        training["linear_density_reference"] = reference
    return Contract(Path("unused-contract.json"), {
        "schema": "shared-xyt-reconstruction-1", "training": training})


def checkpoint(domain="linear", reference=1e12):
    definition = {"data_loss_domain": domain, "linear_density_reference": reference}
    return {"step": 5000, "experiment_contract_sha256": "matching-contract",
            "config": {**definition, "data_loss_definition": definition.copy(),
                       "target_col": "log10_Ne", "num_steps": 5000},
            "target_scaler": {"column": "log10_Ne", "min": 9.5, "max": 12.5}}


def test_omitted_selectors_keep_old_log_contract_and_checkpoint_meaning():
    old = contract()
    assert old.loss_definition() == {"data_loss_domain": "log", "linear_density_reference": None}
    assert old.validate_checkpoint_loss({"step": 5000, "config": {"num_steps": 5000}}) == old.loss_definition()
    # Historical analytic fixtures had no budget/config and only three updates.
    fixture = Contract(Path("unused.json"), {"schema": "shared-xyt-reconstruction-1"})
    assert fixture.validate_checkpoint_loss({"step": 3}) == old.loss_definition()


@pytest.mark.parametrize("domain,reference", [
    ("linear", None), ("linear", 0), ("linear", -1),
    ("linear", float("inf")), ("linear", float("nan")),
    ("log", 1e12), ("unsupported", None),
])
def test_invalid_or_ambiguous_contract_objectives_fail(domain, reference):
    with pytest.raises(ValueError):
        contract(domain, reference).loss_definition()


def test_contract_load_rejects_bad_objective_before_reading_inputs(tmp_path):
    data = {"schema": "shared-xyt-reconstruction-1",
            "roles": {"train": [0], "development": [], "reserved": []},
            "checkpoint_policy": "final_update", "inputs": {"train": {}},
            "training": {"data_loss_domain": "linear", "target_col": "log10_Ne"}}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(data))
    with patch.object(Contract, "file") as file:
        with pytest.raises(ValueError, match="finite positive"):
            Contract.load(path)
    file.assert_not_called()


def test_explicit_metadata_must_match_effective_training_settings():
    case = contract("linear", 1e12)
    case.data["data_loss_definition"] = case.loss_definition()
    assert case.validate_checkpoint_loss(checkpoint()) == case.loss_definition()
    case.data["data_loss_definition"]["linear_density_reference"] = 1e11
    with pytest.raises(ValueError, match="Contract data-loss metadata"):
        case.loss_definition()


@pytest.mark.parametrize("target", [None, "Ne", "density"])
def test_linear_target_must_explicitly_be_log_density(target):
    case = contract("linear", 1e12)
    case.data["training"]["target_col"] = target
    with pytest.raises(ValueError, match="target_col=log10_Ne"):
        case.loss_definition()


@pytest.mark.parametrize("change", [
    {"data_loss_domain": "log", "linear_density_reference": None},
    {"linear_density_reference": 1e11},
    {"data_loss_definition": {"data_loss_domain": "log", "linear_density_reference": None}},
    {"data_loss_definition": None}, {"target_col": "Ne"},
])
def test_checkpoint_objective_cannot_be_relabelled(change):
    saved = checkpoint()
    saved["config"].update(change)
    with pytest.raises(ValueError):
        contract("linear", 1e12).validate_checkpoint_loss(saved)


@pytest.mark.parametrize("scaler", [
    {}, {"column": "Ne", "min": 9.5, "max": 12.5},
    {"column": "log10_Ne", "min": 12.5, "max": 9.5},
    {"column": "log10_Ne", "min": 9.5, "max": 9.5},
    {"column": "log10_Ne", "min": 9.5, "max": float("nan")},
])
def test_checkpoint_linear_target_scaler_must_be_usable(scaler):
    saved = checkpoint()
    saved["target_scaler"] = scaler
    with pytest.raises(ValueError):
        contract("linear", 1e12).validate_checkpoint_loss(saved)


@pytest.mark.parametrize("step,config_budget", [(3, 5000), (4999, 5000), (5000, 4999)])
def test_analysis_requires_requested_final_training_budget(step, config_budget):
    saved = checkpoint()
    saved["step"] = step
    saved["config"]["num_steps"] = config_budget
    with pytest.raises(ValueError, match="requested training budget"):
        contract("linear", 1e12).validate_checkpoint_loss(saved)


def test_three_update_execution_check_keeps_matching_analysis():
    case = contract("linear", 1e12)
    saved = checkpoint()
    saved["step"] = saved["config"]["num_steps"] = 3
    assert case.validate_checkpoint_loss(saved) == case.loss_definition()


def test_explicitly_disabled_target_normalization_is_rejected():
    saved = checkpoint()
    saved["config"]["normalize_values"] = False
    with pytest.raises(ValueError, match="normalized log10_Ne"):
        contract("linear", 1e12).validate_checkpoint_loss(saved)


def test_analysis_rejects_mismatched_objective_before_model_construction(tmp_path):
    case = contract("linear", 1e12)
    saved = checkpoint("log", None)
    with patch("inr_isr.experiments.analysis.ExperimentWindow"), \
         patch("inr_isr.experiments.analysis.resolve_device", return_value="unused-device"), \
         patch("inr_isr.experiments.analysis.digest", return_value="matching-contract"), \
         patch("inr_isr.experiments.analysis.torch_load_checkpoint", return_value=saved), \
         patch("inr_isr.experiments.analysis.build_model_from_checkpoint") as build:
        with pytest.raises(ValueError, match="Checkpoint data-loss definition"):
            analyze(case, tmp_path / "run", tmp_path / "analysis")
    build.assert_not_called()
    assert not (tmp_path / "analysis").exists()
