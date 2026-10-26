"""Non-neural contract/math/analysis tests; no network or optimizer is executed."""
import json
from pathlib import Path
import tempfile
import unittest
import importlib
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from inr_isr.experiments.contract import Contract, digest, grouped_metrics, improves
from inr_isr.experiments.analysis import physical_derivatives, crossings, half_height_interval, width_around_peak, error_metrics, analyze
from inr_isr.experiments.validation import DevelopmentMonitor


class AnalyticPolynomial:
    """Closed-form function of coordinates; no neural layers or parameters."""
    def __call__(self, q):
        return q[:, :1]**2 + 2*q[:, 1:2]**2 + 3*q[:, 2:3]**2 + 4*q[:, :1]*q[:, 1:2]

    def parameters(self):
        return iter(())


class ContractMath(unittest.TestCase):
    def test_macro_score_group_weighting_and_ties(self):
        score = grouped_metrics(np.array([0., 0., 4.]), np.zeros(3), np.array([1, 1, 2]))
        self.assertAlmostEqual(score["macro_rmse_dex"], np.sqrt(8))
        self.assertAlmostEqual(score["pooled_rmse_dex"], np.sqrt(16/3))
        self.assertFalse(improves(2., 2.))
        self.assertTrue(improves(1., 2.))
        with self.assertRaises(ValueError):
            improves(float("nan"), 2.)
        self.assertIsNone(error_metrics(np.array([], dtype=object))["rmse"])
        with self.assertRaises(ValueError):
            error_metrics([float("inf")])

    def test_missing_and_multiple_feature_crossings(self):
        x = np.linspace(-2, 2, 9)
        np.testing.assert_array_equal(crossings(x, x, 0), [0])
        width, count = width_around_peak(x, 1 - abs(x)/2)
        self.assertEqual((width, count), (2., 2))
        self.assertEqual(width_around_peak(x, np.zeros(len(x))), (None, 0))
        # A flat core has many equally valid maxima. Its interval centre stays fixed.
        flat = np.minimum(1., 2 - abs(x))
        left, right, count = half_height_interval(x, flat)
        self.assertEqual((left + right) / 2, 0.)
        self.assertEqual((right - left, count), (3., 2))

    def test_physical_first_and_second_derivative_units(self):
        q = np.array([[.1, -.2, .3], [-.3, .4, -.1]], dtype=np.float32)
        coord = {"x_km": {"min": -5, "max": 5}, "y_km": {"min": 0, "max": 20},
                 "t_sec": {"min": 0, "max": 200}}
        result = physical_derivatives(AnalyticPolynomial(), q, coord, {"min": 9, "max": 13}, torch.device("cpu"), chunk=1)
        np.testing.assert_allclose(result["x"], 2*.2*(2*q[:, 0]+4*q[:, 1]), rtol=2e-7)
        np.testing.assert_allclose(result["y"], 2*.1*(4*q[:, 1]+4*q[:, 0]), rtol=2e-7)
        np.testing.assert_allclose(result["t"], 2*.01*6*q[:, 2], rtol=2e-7)
        for key, expected in (("xx", 2*2*.2**2), ("xy", 2*4*.2*.1), ("yy", 2*4*.1**2), ("tt", 2*6*.01**2)):
            np.testing.assert_allclose(result[key], expected, rtol=2e-7)

    def test_contract_rejects_overlap_and_changed_inputs(self):
        with tempfile.TemporaryDirectory(prefix="xyt-contract-") as tmp:
            root = Path(tmp)
            (root / "input.csv").write_text("a\n1\n")
            path = root / "contract.json"
            c = {"schema": "shared-xyt-development-1", "roles": {"train": [1], "development": [2], "reserved": [3]},
                 "inputs": {"train": {"path": "input.csv", "sha256": digest(root / "input.csv")}}}
            path.write_text(json.dumps(c))
            Contract.load(path)
            c["roles"]["reserved"] = [1]
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                Contract.load(path)
            c["roles"]["reserved"] = [3]
            path.write_text(json.dumps(c))
            (root / "input.csv").write_text("a\n2\n")
            with self.assertRaises(ValueError):
                Contract.load(path)

    def test_selection_uses_updated_state_and_keeps_earlier_tie(self):
        from synthetic_dataset import SyntheticPlasmaTimeDataset

        class ConstantFunction:
            training = True
            value = .25

            def __call__(self, q):
                return torch.full((len(q), 1), self.value, device=q.device)

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def state_dict(self):
                return {"analytic_value": self.value}

        with tempfile.TemporaryDirectory(prefix="xyt-selection-function-") as tmp:
            root = Path(tmp)
            train = pd.DataFrame({"x_km": [0., 1.], "y_km": [0., 1.], "t_sec": [0., 1.],
                                  "time_index": [0, 1], "log10_Ne": [9.5, 11.5], "beam_id": [1, 1]})
            train.to_csv(root / "train.csv", index=False)
            train.assign(beam_id=2, log10_Ne=10.75).to_csv(root / "development.csv", index=False)
            train[["x_km", "y_km", "t_sec"]].to_csv(root / "probes.csv", index=False)
            inputs = {name: {"path": f"{name}.csv", "sha256": digest(root / f"{name}.csv")}
                      for name in ("train", "development", "probes")}
            path = root / "contract.json"
            path.write_text(json.dumps({"schema": "shared-xyt-development-1", "id": "analytic-function-test",
                "roles": {"train": [1], "development": [2], "reserved": [3]},
                "selection_every": 100, "prediction_chunk": 1, "inputs": inputs}))
            dataset = SyntheticPlasmaTimeDataset(root / "train.csv", verbose=False)
            output = root / "run"
            output.mkdir()
            monitor = DevelopmentMonitor(Contract.load(path), dataset, output)
            function = ConstantFunction()
            before = torch.random.get_rng_state().clone()
            monitor.diagnostic_points(torch.device("cpu"))
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            monitor.after_update(99, 350, function, {}, torch.device("cpu"))
            self.assertFalse((output / "model_best_development.pt").exists())
            monitor.after_update(100, 350, function, {}, torch.device("cpu"))
            monitor.after_update(200, 350, function, {}, torch.device("cpu"))
            function.value = 0.
            monitor.after_update(350, 350, function, {}, torch.device("cpu"))
            selected = torch.load(output / "model_best_development.pt", weights_only=False)
            self.assertEqual(selected["step"], 100)
            self.assertEqual(selected["model_state_dict"]["analytic_value"], .25)
            self.assertEqual(monitor.history[-1]["step"], 350)
            self.assertTrue(function.training)

    def test_stage_receipt_records_runtime_and_failure_without_neural_dispatch(self):
        entry = importlib.import_module("inr_isr.experiments.__main__")
        with tempfile.TemporaryDirectory(prefix="xyt-stage-receipt-") as tmp:
            root = Path(tmp)
            for name, failure in (("success", False), ("failure", True)):
                output = root / name
                def dispatch(*args):
                    if failure:
                        raise RuntimeError("intentional function-only failure")
                    output.mkdir()
                with patch.object(sys, "argv", ["experiment", "train", "--contract", str(root / "dummy.json"), "--output", str(output)]), \
                     patch.object(entry.Contract, "load", return_value=object()), \
                     patch.object(entry, "train", side_effect=dispatch), \
                     patch.object(entry, "runtime_identity", return_value={"interpreter": sys.executable}):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "intentional"):
                            entry.main()
                    else:
                        entry.main()
                receipt = json.loads((output / "stage_receipt.json").read_text())
                self.assertEqual(receipt["status"], "failed" if failure else "completed")
                self.assertEqual(receipt["runtime"]["interpreter"], sys.executable)
                self.assertIn("started_utc", receipt)
                self.assertIn("ended_utc", receipt)

    def test_full_analysis_with_closed_form_function_only(self):
        with tempfile.TemporaryDirectory(prefix="xyt-analysis-function-") as tmp:
            root = Path(tmp)
            scene = {"schema_version": 2, "duration_sec": 43200, "truth_margin_sec": 300,
                "front": {"low_log10_ne": 9.5, "high_log10_ne": 11.5, "center_y_km": 255,
                    "amplitude_km": 60, "wavelength_km": 720, "vertical_width_10_90_km": 180, "phase_rad": 0},
                "patches": [{"name": "p1", "pair_id": "pair", "region": "low", "edge_class": "sharp",
                    "x0_km": -2000, "y0_km": 100, "half_height_diameter_km": 300,
                    "edge_width_10_90_km": 40, "enhancement_dex": 1}]}
            (root / "scene.json").write_text(json.dumps(scene))
            observations = pd.DataFrame({"x_km": [0., 100.], "y_km": [100., 300.], "t_sec": [3600., 14400.],
                                         "log10_Ne": [9.5, 11.5], "beam_id": [1, 1]})
            observations.to_csv(root / "train.csv", index=False)
            observations.assign(beam_id=2).to_csv(root / "development.csv", index=False)
            queries = observations[["x_km", "y_km", "t_sec"]].assign(inside_training_hull=[True, False], nearest_training_km=[1., 100.])
            queries.to_csv(root / "queries.csv", index=False)
            pd.DataFrame({"x_km": [0., 100., 200.], "y_km": [100., 300., 150.],
                          "role": ["train", "development", "reserved"]}).to_csv(root / "roles.csv", index=False)
            path = root / "contract.json"
            settings = {"derivative_query_count": 2, "patch_threshold_dex": .1, "front_fraction_bounds": [.1, .9],
                        "grid_nx": 8, "grid_ny": 9, "map_times_sec": [3605], "cut_y_km": 255}
            inputs = {name: {"path": f"{name}.{'json' if name=='scene' else 'csv'}",
                       "sha256": digest(root / f"{name}.{'json' if name=='scene' else 'csv'}")}
                      for name in ("scene", "train", "development", "queries", "roles")}
            path.write_text(json.dumps({"schema": "shared-xyt-development-1", "case": "analytic-function-test",
                "roles": {"train": [1], "development": [2], "reserved": [3]}, "inputs": inputs,
                "speed_km_s": .4, "prediction_chunk": 1024, "analysis": settings}))
            contract = Contract.load(path)
            run = root / "run"
            run.mkdir()
            (run / "model_best_development.pt").write_text("Test marker; not a neural checkpoint")
            scalers = {c: {"min": lo, "max": hi} for c, lo, hi in
                       (("x_km", -100, 400), ("y_km", 0, 510), ("t_sec", 3600, 14400))}
            checkpoint = {"coord_scalers": scalers, "target_scaler": {"min": 9.5, "max": 12.5},
                          "step": 3, "experiment_contract_sha256": digest(path)}
            pd.DataFrame({"step": [1, 3], "data_loss": [1, .5],
                **{key: [1., 1.] for key in ("fxx_rms", "fxy_rms", "fyy_rms", "ftt_rms")}}).to_csv(run / "history.csv", index=False)
            pd.DataFrame({"step": [3], "macro_rmse_dex": [1.]}).to_csv(run / "development_history.csv", index=False)
            with patch("inr_isr.experiments.analysis.torch_load_checkpoint", return_value=checkpoint), \
                 patch("inr_isr.experiments.analysis.build_model_from_checkpoint", return_value=AnalyticPolynomial()):
                analyze(contract, run, root / "analysis", cpu=True)
            summary = json.loads((root / "analysis/summary.json").read_text())
            self.assertEqual(summary["features"]["patch_profiles"], 1)
            self.assertGreater(summary["features"]["missing_front_crossings"], 0)
            self.assertEqual(summary["regions"][0]["count"], 2)
            self.assertTrue((root / "analysis/derivative_queries_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
