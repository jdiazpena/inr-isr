"""Optional post-update selection; native optimization/controller stages stay intact."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inr_isr.prediction import predict_log10_on_coords
from inr_isr.transforms import normalize_coords_from_scalers
from .contract import digest, grouped_metrics, improves
from .window import ExperimentWindow


class DevelopmentMonitor:
    def __init__(self, contract, dataset, output):
        self.contract, self.dataset, self.output = contract, dataset, Path(output)
        self.window = ExperimentWindow.from_contract(contract).validate_inputs(contract)
        self.window.assert_times(dataset.df.t_sec, "selected training data")
        require_endpoints = contract.data.get('window_policy', {}).get('require_observed_endpoints', False)
        self.window.assert_scalers(dataset.coord_scalers, require_endpoints)
        self.development = pd.read_csv(contract.file("development"))
        expected = set(contract.data["roles"]["development"])
        if set(self.development.beam_id.unique()) != expected:
            raise ValueError("Development table does not match its declared beam roles")
        if set(dataset.df.beam_id.unique()) != set(contract.data["roles"]["train"]):
            raise ValueError("Training table does not match its declared beam roles")
        if digest(dataset.csv_path) != contract.data["inputs"]["train"]["sha256"]:
            raise ValueError("Dataset differs from contracted training input")
        self.coordinates = normalize_coords_from_scalers(
            *[self.development[c].to_numpy() for c in ("x_km", "y_km", "t_sec")], dataset.coord_scalers)
        self.best_score, self.best_step, self.latest = float("inf"), None, None
        self.history = []
        self.output.joinpath("experiment_contract.json").write_text(json.dumps(contract.data, indent=2) + "\n")
        self.output.joinpath("preprocessing.json").write_text(json.dumps({
            "policy": "fit all coordinate and target limits on supplied training rows only; no clipping",
            "train_rows": dataset.n_samples, "coord_scalers": dataset.coord_scalers,
            "target_scaler": dataset.target_scaler}, indent=2) + "\n")

    def diagnostic_points(self, device):
        probes = pd.read_csv(self.contract.file("probes"))
        probes.to_csv(self.output / "diagnostic_points_physical.csv", index=False)
        coordinates = normalize_coords_from_scalers(
            *[probes[c].to_numpy() for c in ("x_km", "y_km", "t_sec")], self.dataset.coord_scalers)
        # Own storage, independent of model RNG. These points do not enter the data loss.
        return torch.from_numpy(coordinates.copy()).to(device)

    def after_update(self, step, total_steps, model, config, device):
        every = self.contract.data["selection_every"]
        if step % every and step != total_steps:
            return
        training = model.training
        model.eval()
        try:
            prediction = predict_log10_on_coords(model, self.coordinates, self.dataset.target_scaler,
                                                 device, self.contract.data["prediction_chunk"])
            score = grouped_metrics(prediction, self.development.log10_Ne.to_numpy(),
                                    self.development.beam_id.to_numpy())
        finally:
            model.train(training)
        self.latest = score["macro_rmse_dex"]
        if improves(self.latest, self.best_score):
            self.best_score, self.best_step = self.latest, step
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                "coord_scalers": self.dataset.coord_scalers, "target_scaler": self.dataset.target_scaler,
                "checkpoint_type": "best_development_after_update", "step": step,
                "development_score": score, "experiment_id": self.contract.data["id"],
                "experiment_contract_sha256": digest(self.contract.path)},
                self.output / "model_best_development.pt")
        self.history.append({"step": step, "macro_rmse_dex": self.latest,
                             "pooled_rmse_dex": score["pooled_rmse_dex"],
                             "best_step": self.best_step, "best_macro_rmse_dex": self.best_score})
        pd.DataFrame(self.history).to_csv(self.output / "development_history.csv", index=False)
        (self.output / "selection.json").write_text(json.dumps({"best_step": self.best_step,
            "best_macro_rmse_dex": self.best_score, "completed_updates": step,
            "rule": "minimum post-update macro RMSE in dex; earlier exact tie wins"}, indent=2) + "\n")


class ReconstructionMonitor(DevelopmentMonitor):
    """Fixed probes and final-update saving; no prediction-based model selection."""

    def __init__(self, contract, dataset, output):
        if not contract.all_beam:
            raise ValueError("ReconstructionMonitor requires an explicit all-beam contract")
        contract.validate_all_beam_inputs()
        self.contract, self.dataset, self.output = contract, dataset, Path(output)
        self.window = ExperimentWindow.from_contract(contract).validate_inputs(contract)
        self.window.assert_times(dataset.df.t_sec, "all-beam training data")
        self.window.assert_scalers(dataset.coord_scalers, True)
        if digest(dataset.csv_path) != contract.data["inputs"]["train"]["sha256"]:
            raise ValueError("Dataset differs from the all-beam training input")
        if dataset.n_samples != len(pd.read_csv(contract.file("train"))):
            raise ValueError("Dataset silently removed observations")
        self.latest = self.best_step = None
        self.output.joinpath("experiment_contract.json").write_text(json.dumps(contract.data, indent=2) + "\n")
        self.output.joinpath("preprocessing.json").write_text(json.dumps({
            "policy": "fit all coordinate and target limits on all supplied measurements; no clipping",
            "train_rows": dataset.n_samples, "coord_scalers": dataset.coord_scalers,
            "target_scaler": dataset.target_scaler}, indent=2) + "\n")

    def after_update(self, step, total_steps, model, config, device):
        if step != total_steps:
            return
        torch.save({"model_state_dict": model.state_dict(), "config": config,
            "coord_scalers": self.dataset.coord_scalers, "target_scaler": self.dataset.target_scaler,
            "checkpoint_type": "final_reconstruction_after_update", "step": step,
            "experiment_id": self.contract.data["id"],
            "experiment_contract_sha256": digest(self.contract.path)},
            self.output / "model_reconstruction.pt")
        (self.output / "selection.json").write_text(json.dumps({
            "completed_updates": step, "rule": "final_update", "step": step,
            "withheld_beams": 0, "withheld_times": 0, "score_based_selection": False}, indent=2) + "\n")
