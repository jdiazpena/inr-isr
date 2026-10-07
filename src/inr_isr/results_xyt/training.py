"""User-executed CUDA training for the explicit xyt reconstruction protocol.

No model is constructed at import. Neural execution is confined to these named
functions, so arithmetic/data checks can run without entering this module.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
import numpy as np
import torch

from .common import fresh_directory, sha256, write_json
from .config import training_settings
from .data import exposure_points, load_prepared, normalize_points
from .math import metrics, tensor_observation_log10, tensor_observation_mse


@torch.no_grad()
def predict_observations(model, rows, scaler, bounds, device, microbatch=128):
    model.eval()
    outputs = []
    for start in range(0, len(rows), microbatch):
        group = rows.iloc[start:start + microbatch]
        coordinates = normalize_points(exposure_points(group).reshape(-1, 3), bounds)
        points = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
        nodes = model(points).reshape(len(group), 33)
        values = tensor_observation_log10(nodes, scaler)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite integrated prediction")
        outputs.append(values.cpu().numpy().astype(np.float64))
    return np.concatenate(outputs)


@torch.no_grad()
def predict_field(model, coordinates, scaler, bounds, device, chunk_size=4096):
    """Reuse the existing ordered prediction kernel for instantaneous queries."""
    from inr_isr.prediction import predict_log10_on_coords
    model.eval()
    values = predict_log10_on_coords(model, normalize_points(coordinates, bounds), scaler, device, chunk_size)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite field prediction")
    return values


def _save_checkpoint(path, model, optimizer, state):
    # Live best/last updates only; the enclosing output directory must be fresh.
    temporary = path.with_name(path.name + ".partial")
    torch.save({**state, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict()}, temporary)
    temporary.replace(path)


def train(case, config_identity, data_path, output, smoke=False):
    """One full-training-mean Adam update assembled from observation microbatches."""
    from models import MLPINR
    from inr_isr.device import resolve_device, device_record
    from training_common import set_seed
    settings = training_settings(case, smoke)
    device = resolve_device(settings["device"])  # No fallback; before model/data/output work.
    rows, data_record = load_prepared(data_path, config_identity)
    training = rows[rows.role == "train"].copy()
    validation = rows[rows.role == "validation"].copy()
    scaler, bounds = data_record["scaler"], data_record["bounds"]
    if not len(training) or not len(validation):
        raise ValueError("Both training and validation must be nonempty")
    output = fresh_directory(output)
    write_json(output / "effective_training.json", settings)
    set_seed(settings["seed"])
    model = MLPINR(in_features=3, out_features=1, **settings["model"]).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    coordinates = normalize_points(exposure_points(training).reshape(-1, 3), bounds).reshape(len(training), 33, 3)
    values = training.log10_Ne.to_numpy()
    initial = torch.as_tensor(coordinates[:1].reshape(-1, 3), device=device)
    execution = device_record(settings["device"], device, model=model, batch=initial, amp=False)
    initial_target = torch.as_tensor(values[:1], dtype=torch.float32, device=device)
    execution["target_device"] = str(initial_target.device)
    write_json(output / "execution_device.json", execution)
    state = {"format": "xyt_results_checkpoint_v1", "config_identity": config_identity,
             "data_manifest_sha256": sha256(Path(data_path) / "data_manifest.json"),
             "settings": settings, "target_scaler": scaler, "bounds": bounds,
             "fit_observation_ids": training.observation_id.tolist(),
             "validation_observation_ids": validation.observation_id.tolist(),
             "final_targets_used": False, "quadrature_nodes": 33, "smoke": smoke}
    best_score, best_step, completed = float("inf"), None, 0
    last_validation, last_validated_step, last_loss = None, None, None
    started = time.monotonic()
    fields = ["event", "step", "training_normalized_mse", "validation_macro_log10_rmse", "elapsed_training_seconds", "grad_norm",
              "lambda_curv_xy_eff", "lambda_curv_t_eff", "reference_loss_weights", "training_rows", "validation_rows"]
    with (output / "history.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def validate_and_save(step):
            nonlocal best_score, best_step, last_validation, last_validated_step
            prediction = predict_observations(model, validation, scaler, bounds, device, settings["microbatch_observations"])
            summary, _ = metrics(validation.log10_Ne.to_numpy(), prediction, validation.beam_index.to_numpy())
            score = summary["log10_macro"]["rmse"]
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation metric")
            last_validation, last_validated_step = score, step
            if score < best_score:  # Exact ties keep the earlier saved checkpoint.
                best_score, best_step = score, step
                _save_checkpoint(output / "best_validation.pt", model, optimizer,
                                 {**state, "step": step, "validation_macro_log10_rmse": score})
            print(f"step={step} validation_macro_log10_rmse={score:.6g} best_step={best_step}", flush=True)
            return score

        def history(event, step, loss, grad_norm):
            writer.writerow(dict(event=event, step=step, training_normalized_mse=loss,
                                 validation_macro_log10_rmse=last_validation if last_validated_step == step else "",
                                 elapsed_training_seconds=time.monotonic() - started, grad_norm=grad_norm,
                                 lambda_curv_xy_eff=0.0, lambda_curv_t_eff=0.0, reference_loss_weights=False,
                                 training_rows=len(training), validation_rows=len(validation)))
            handle.flush()

        for step in range(1, settings["max_updates"] + 1):
            if time.monotonic() - started >= settings["max_seconds"]:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            full_mean = 0.0
            for start in range(0, len(training), settings["microbatch_observations"]):
                stop = min(start + settings["microbatch_observations"], len(training))
                points = torch.as_tensor(coordinates[start:stop].reshape(-1, 3), dtype=torch.float32, device=device)
                target = torch.as_tensor(values[start:stop], dtype=torch.float32, device=device)
                nodes = model(points).reshape(stop - start, 33)
                loss = tensor_observation_mse(nodes, target, scaler) * ((stop - start) / len(training))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                full_mean += float(loss.detach().item())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Missing/nonfinite parameter gradient")
            grad_norm = float(torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters())).item())
            if not np.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            completed, last_loss = step, full_mean
            if step % settings["validation_every"] == 0:
                validate_and_save(step)
            history("update", step, full_mean, grad_norm)
        if not completed:
            raise RuntimeError("Budget elapsed before an optimizer update completed")
        if last_validated_step != completed:
            validate_and_save(completed)
            history("final_validation", completed, last_loss, "")
    _save_checkpoint(output / "last.pt", model, optimizer,
                     {**state, "step": completed, "validation_macro_log10_rmse": last_validation})
    result = {"completed_updates": completed, "best_validation_step": best_step,
              "best_validation_macro_log10_rmse": best_score, "training_seconds": time.monotonic() - started,
              "stop_reason": "update_cap" if completed == settings["max_updates"] else "wall_cap",
              "budget_scope": "Optimizer loop including validation; check at update boundaries, final validation/save may extend wall cap",
              "training_rows": len(training), "validation_rows": len(validation), "final_targets_used": False,
              "best_checkpoint_sha256": sha256(output / "best_validation.pt"), "last_checkpoint_sha256": sha256(output / "last.pt")}
    write_json(output / "training_result.json", result)
    return result


def load_model(checkpoint_path, config_identity, data_path, device_policy):
    from models import MLPINR
    from inr_isr.device import resolve_device
    device = resolve_device(device_policy)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("format") != "xyt_results_checkpoint_v1" or state["config_identity"] != config_identity or state["final_targets_used"]:
        raise ValueError("Checkpoint format/case/role mismatch")
    if state["data_manifest_sha256"] != sha256(Path(data_path) / "data_manifest.json"):
        raise ValueError("Checkpoint prepared-data binding mismatch")
    model = MLPINR(in_features=3, out_features=1, **state["settings"]["model"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, state, device
