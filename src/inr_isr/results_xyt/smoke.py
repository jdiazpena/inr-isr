"""NEURAL checks for Joaquin only; Codex must not execute this module's verify."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .common import fresh_directory, write_json
from .data import exposure_points, load_prepared, normalize_points
from .math import metrics, tensor_observation_mse
from .training import load_model, predict_observations


def verify(case, identity, data, checkpoint, output):
    """Check real CUDA gradients, microbatch weighting and actual best selection.

The float32 comparison allows accumulation/reduction rounding: atol1e-6,
rtol2e-4. This tests one mathematical objective, not old-training parity or a
scientific accuracy threshold. Criteria are declared before user execution.
"""
    rows, manifest = load_prepared(data, identity)
    model, state, device = load_model(checkpoint, identity, data, case["training"]["device"])
    if not state["smoke"] or device.type != "cuda":
        raise ValueError("Neural smoke verification requires its CUDA smoke checkpoint")
    history = pd.read_csv(Path(checkpoint).parent / "history.csv")
    candidates = history[history.validation_macro_log10_rmse.notna()]
    selected = candidates.loc[candidates.validation_macro_log10_rmse.idxmin()]
    if int(selected.step) != state["step"] or float(selected.validation_macro_log10_rmse) != state["validation_macro_log10_rmse"]:
        # CSV decimal roundtrip is required; parse exactly below before rejecting.
        history = pd.read_csv(Path(checkpoint).parent / "history.csv", float_precision="round_trip")
        candidates = history[history.validation_macro_log10_rmse.notna()]
        selected = candidates.loc[candidates.validation_macro_log10_rmse.idxmin()]
        if int(selected.step) != state["step"] or float(selected.validation_macro_log10_rmse) != state["validation_macro_log10_rmse"]:
            raise AssertionError("Saved checkpoint does not match the recorded best validation selection")
    if state["fit_observation_ids"] != rows[rows.role == "train"].observation_id.tolist():
        raise AssertionError("Checkpoint trained on the wrong rows")
    if state["validation_observation_ids"] != rows[rows.role == "validation"].observation_id.tolist():
        raise AssertionError("Checkpoint validation roles are wrong")
    validation = rows[rows.role == "validation"]
    recomputed = predict_observations(model, validation, state["target_scaler"], state["bounds"], device)
    actual_score = metrics(validation.log10_Ne.to_numpy(), recomputed, validation.beam_index.to_numpy())[0]["log10_macro"]["rmse"]
    if not np.isclose(actual_score, state["validation_macro_log10_rmse"], atol=1e-7, rtol=1e-6):
        raise AssertionError("Reloaded checkpoint does not reproduce its full validation score")
    result_training = json.loads((Path(checkpoint).parent / "training_result.json").read_text())
    last = torch.load(Path(checkpoint).parent / "last.pt", map_location="cpu", weights_only=True)
    if result_training["completed_updates"] != 2 or last["step"] != 2 or last["final_targets_used"]:
        raise AssertionError("Smoke must complete two updates and keep a separate last checkpoint")
    group = rows[rows.role == "train"].iloc[:7]
    nodes = torch.as_tensor(normalize_points(exposure_points(group).reshape(-1, 3), state["bounds"]), device=device)
    targets = torch.as_tensor(group.log10_Ne.to_numpy(), dtype=torch.float32, device=device)
    parameters = tuple(model.parameters())
    full = tensor_observation_mse(model(nodes).reshape(7, 33), targets, state["target_scaler"])
    whole = torch.autograd.grad(full, parameters)
    accumulated = [torch.zeros_like(p) for p in parameters]
    for a, b in ((0, 3), (3, 6), (6, 7)):
        loss = tensor_observation_mse(model(nodes[a * 33:b * 33]).reshape(b - a, 33), targets[a:b], state["target_scaler"]) * ((b - a) / 7)
        for total, gradient in zip(accumulated, torch.autograd.grad(loss, parameters)):
            total.add_(gradient)
    if not all(torch.isfinite(g).all() for g in whole) or not any(torch.any(g != 0) for g in whole):
        raise AssertionError("No finite nonzero gradient reached real model parameters")
    for actual, expected in zip(accumulated, whole):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-4)
    execution = json.loads((Path(checkpoint).parent / "execution_device.json").read_text())
    if not all(str(execution[k]).startswith("cuda") for k in ("model_device", "batch_device", "target_device")) or execution["amp"] or execution["dtype"] != "torch.float32":
        raise AssertionError("Smoke did not use the declared CUDA/no-AMP policy")
    if manifest["final_targets_generated"] or state["final_targets_used"]:
        raise AssertionError("Reserved final targets were consumed")
    output = fresh_directory(output)
    result = {"status": "passed", "config_identity": identity, "real_model_gradient": True,
              "microbatch_gradient_equivalence": True, "best_validation_selection": True,
              "reloaded_validation_score": actual_score, "separate_last_checkpoint": True,
              "device": str(device), "final_targets_used": False,
              "gradient_atol": 1e-6, "gradient_rtol": 2e-4,
              "limitation": "Two-update functionality check, not reconstruction accuracy or convergence"}
    write_json(output / "neural_checks.json", result)
    return result
