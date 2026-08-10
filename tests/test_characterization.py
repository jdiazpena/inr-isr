"""Numerical contracts that must remain stable during the readability refactor."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from amisr_h5_reader_3d import read_amisr_h5_3d_altitude_band
from datasets import RadarTimeH5Dataset
from models import MLPINR
from synthetic_plasma import (
    MovingGaussianPatch,
    evaluate_integration_averaged_plasma,
    evaluate_synthetic_plasma,
    make_observation_geometry,
)
from synthetic_train_3d import build_parser as build_synthetic_train_parser
from synthetic_train_3d import curvature_losses_xy_t
from training_config import documented_defaults, parse_args_with_optional_json
import synthetic_analyze_reconstruction as canonical_analysis
import synthetic_analyze_reconsturction_linear_errors as historical_analysis
from train_radar_3d_window_reference_reg_diagnostic import (
    build_parser as build_radar_train_parser,
)
from run_reference_windows import build_command as build_reference_window_command
from run_reference_windows_diagnostics_minimal import (
    build_command as build_diagnostic_window_command,
)


class QuadraticField(torch.nn.Module):
    """Analytical field with known normalized-coordinate second derivatives."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        return x**2 + x * y + y**2 + t**2


class ModelCharacterizationTest(unittest.TestCase):
    def test_historical_analysis_name_aliases_canonical_module(self) -> None:
        self.assertIs(historical_analysis, canonical_analysis)

    def test_seeded_siren_output_is_stable(self) -> None:
        torch.manual_seed(123)
        model = MLPINR(
            in_features=3,
            out_features=1,
            hidden_features=8,
            hidden_layers=1,
            activation="sine",
            first_omega_0=5.0,
            hidden_omega_0=5.0,
        )
        coords = torch.tensor(
            [[-1.0, 0.0, 1.0], [0.25, -0.5, 0.75], [1.0, 1.0, -1.0]],
            dtype=torch.float32,
        )
        expected = np.array(
            [-0.08823274075984955, -0.013220027089118958, -0.017034776508808136]
        )
        np.testing.assert_allclose(model(coords).detach().numpy()[:, 0], expected, rtol=0, atol=1e-7)

    def test_curvature_equations_are_unchanged(self) -> None:
        coords = torch.tensor(
            [[-0.5, 0.2, 0.1], [0.0, -0.3, 0.8], [0.9, 0.4, -0.7]],
            dtype=torch.float32,
        )
        spatial, temporal = curvature_losses_xy_t(QuadraticField(), coords, True, True)
        self.assertAlmostEqual(float(spatial.detach()), 10.0, places=6)
        self.assertAlmostEqual(float(temporal.detach()), 4.0, places=6)

    def test_synthetic_and_radar_training_defaults_match(self) -> None:
        synthetic = build_synthetic_train_parser().parse_args([])
        radar = build_radar_train_parser().parse_args([])
        shared_names = (
            "activation",
            "hidden_features",
            "hidden_layers",
            "first_omega_0",
            "hidden_omega_0",
            "lr",
            "batch_size",
            "num_steps",
            "seed",
            "lambda_curv_xy",
            "lambda_curv_t",
            "target_xy_ratio",
            "target_t_ratio",
            "epsilon_data",
            "loss_ema_beta",
            "lambda_smoothing",
            "lambda_update_every",
            "lambda_warmup_steps",
            "lambda_curv_xy_max",
            "lambda_curv_t_max",
            "num_collocation",
            "collocation_grid_nx",
            "collocation_grid_ny",
            "reg_ramp_frac",
        )
        self.assertEqual(
            {name: getattr(synthetic, name) for name in shared_names},
            {name: getattr(radar, name) for name in shared_names},
        )

    def test_documented_json_defaults_match_typed_defaults(self) -> None:
        path = PROJECT_ROOT / "config" / "training_defaults.json"
        with path.open("r", encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(stored, documented_defaults())

    def test_command_line_overrides_optional_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.json"
            path.write_text('{"num_steps": 123, "hidden_features": 64}', encoding="utf-8")
            args = parse_args_with_optional_json(
                build_synthetic_train_parser(),
                ["--config", str(path), "--num_steps", "7"],
            )
        self.assertEqual(args.num_steps, 7)
        self.assertEqual(args.hidden_features, 64)
        self.assertFalse(hasattr(args, "config"))


class SyntheticPhysicsCharacterizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.patch = MovingGaussianPatch(
            name="p",
            amplitude_m3=2.0e11,
            sigma_x_km=40.0,
            sigma_y_km=60.0,
            x0_km=-20.0,
            y0_km=10.0,
            vx_km_s=0.4,
            vy_km_s=-0.2,
        )

    def test_analytical_field_and_derivatives(self) -> None:
        values = evaluate_synthetic_plasma(
            np.array([-20.0, 0.0, 50.0]),
            np.array([10.0, -10.0, 25.0]),
            np.array([0.0, 20.0, 100.0]),
            background_ne_m3=1.0e9,
            patches=[self.patch],
        )
        np.testing.assert_allclose(
            values["Ne"],
            [201000000000.0, 185520729280.36514, 128348634335.68192],
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            values["true_dlog10Ne_dt_sec"],
            [0.0, 0.0016798192963800576, 0.0023939487360675198],
            rtol=1e-13,
            atol=1e-15,
        )

    def test_integration_averaging_contract(self) -> None:
        values = evaluate_integration_averaged_plasma(
            np.array([0.0]),
            np.array([0.0]),
            np.array([100.0]),
            integration_time_sec=120.0,
            integration_samples=21,
            background_ne_m3=1.0e9,
            patches=[self.patch],
        )
        np.testing.assert_allclose(values["Ne"], [166966683559.78754], rtol=1e-13)
        np.testing.assert_allclose(
            values["true_dlog10Ne_dt_sec"], [-0.0020960433840745292], rtol=1e-13
        )

    def test_named_beam_geometries(self) -> None:
        expected = {
            "sparse_grid": (42, 1530666.6666666667),
            "sparse_23": (23, 840000.0),
            "sparse_11": (11, 500000.0),
        }
        for mode, (count, squared_radius_sum) in expected.items():
            geometry = make_observation_geometry(mode, domain_size_km=500.0, seed=0)
            self.assertEqual(len(geometry), count)
            self.assertAlmostEqual(float(geometry["x_km"].sum()), 0.0, places=9)
            self.assertAlmostEqual(float(geometry["y_km"].sum()), 0.0, places=9)
            self.assertAlmostEqual(
                float((geometry["x_km"] ** 2 + geometry["y_km"] ** 2).sum()),
                squared_radius_sum,
                places=6,
            )


class RealRadarCharacterizationTest(unittest.TestCase):
    @staticmethod
    def _write_minimal_amisr_file(path: Path) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "BeamCodes",
                data=np.array([[1001.0, 0.0, 45.0], [1002.0, 90.0, 45.0]]),
            )
            fitted = handle.create_group("FittedParams")
            fitted.create_dataset(
                "Ne",
                data=np.array(
                    [
                        [[1.0e10, 2.0e10, 3.0e10], [4.0e10, 5.0e10, 6.0e10]],
                        [[2.0e10, 3.0e10, 4.0e10], [5.0e10, 6.0e10, 7.0e10]],
                    ]
                ),
            )
            fitted.create_dataset(
                "dNe",
                data=np.full((2, 2, 3), 1.0e9, dtype=np.float64),
            )
            fitted.create_dataset(
                "Range",
                data=np.array([[400000.0, 466690.0, 530000.0], [400000.0, 466690.0, 530000.0]]),
            )
            fitted.create_dataset(
                "Altitude",
                data=np.array([[282843.0, 330000.0, 374767.0], [282843.0, 330000.0, 374767.0]]),
            )
            time = handle.create_group("Time")
            time.create_dataset("UnixTime", data=np.array([[1000.0, 1060.0], [1120.0, 1180.0]]))

    def test_reader_and_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal_amisr.h5"
            self._write_minimal_amisr_file(path)
            frame = read_amisr_h5_3d_altitude_band(
                path,
                h0_km=330.0,
                half_width_km=5.0,
                verbose=False,
            )
            self.assertEqual(len(frame), 4)
            self.assertEqual(frame["time_index"].tolist(), [0, 0, 1, 1])
            np.testing.assert_allclose(frame["t_sec"], [0.0, 0.0, 120.0, 120.0])
            np.testing.assert_allclose(frame["Ne"], [2.0e10, 5.0e10, 3.0e10, 6.0e10])
            dataset = RadarTimeH5Dataset(
                path,
                h0_km=330.0,
                half_width_km=5.0,
                verbose=False,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["coords"].shape), (4, 3))
            self.assertEqual(tuple(sample["values"].shape), (4, 1))
            self.assertTrue(torch.isfinite(sample["coords"]).all())
            self.assertTrue(torch.isfinite(sample["values"]).all())

    def test_window_wrappers_generate_valid_trainer_commands(self) -> None:
        common = dict(
            h5_path="example.h5",
            window_size_records=31,
            target_xy_ratio=0.3,
            target_t_ratio=0.3,
            epsilon_data=1.0e-6,
            num_steps=20_000,
            seed=0,
            num_collocation=16_384,
            collocation_grid_nx=80,
            collocation_grid_ny=80,
            freeze_lambdas_after_step=0,
            no_plots=True,
        )
        basic = build_reference_window_command(
            Namespace(**common),
            window_start=15,
            output_dir=Path("outputs/basic"),
        )
        diagnostic = build_diagnostic_window_command(
            Namespace(
                **common,
                deriv_zero_epsilon=1.0e-10,
                num_diagnostic_collocation=8192,
                component_grad_every=500,
            ),
            window_start=15,
            output_dir=Path("outputs/diagnostic"),
        )
        parser = build_radar_train_parser()
        parsed_basic = parser.parse_args(basic[2:])
        parsed_diagnostic = parser.parse_args(diagnostic[2:])
        self.assertTrue(parsed_basic.reference_loss_weights)
        self.assertTrue(parsed_diagnostic.reference_loss_weights)
        self.assertEqual(parsed_basic.h5_path, "example.h5")
        self.assertEqual(parsed_diagnostic.window_start_index, 15)


if __name__ == "__main__":
    unittest.main()
