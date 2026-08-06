# -*- coding: utf-8 -*-
"""Orchestrate the synthetic velocity, integration-time, and shear benchmark."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
BENCHMARK_CONFIG_PATH = PROJECT_ROOT / "config" / "velocity_integration_benchmark.json"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


@dataclass(frozen=True)
class DataCase:
    case_id: str
    motion: str
    geometry: str
    speed_km_s: float
    direction_deg: float
    integration_time_sec: int
    seed: int
    duration_sec: int = 1200

    @property
    def n_times(self) -> int:
        return int(self.duration_sec // self.integration_time_sec) + 1

    @property
    def integration_samples(self) -> int:
        # Resolve the fastest displacement at roughly one eighth of a 45 km patch width.
        samples = max(21, int(math.ceil(self.speed_km_s * self.integration_time_sec / 5.625)) + 1)
        samples = min(samples, 241)
        return samples if samples % 2 == 1 else samples + 1


with BENCHMARK_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    BENCHMARK_CONFIG = json.load(config_file)

REGULARIZATIONS: dict[str, list[str]] = BENCHMARK_CONFIG["regularizations"]


def ordinary_case(
    geometry: str,
    speed: float,
    integration: int,
    seed: int,
    direction: float = 0.0,
) -> DataCase:
    beam_label = {"sparse_grid": "b42", "sparse_23": "b23", "sparse_11": "b11"}[geometry]
    case_id = (
        f"single_{beam_label}_v{speed:.2f}_dir{direction:03.0f}_"
        f"int{integration // 60:02d}m_s{seed}"
    ).replace(".", "p")
    return DataCase(case_id, "left_right", geometry, speed, direction, integration, seed)


def shear_case(seed: int) -> DataCase:
    return DataCase(
        case_id=f"flow_reversal_b23_v1p00_int02m_s{seed}",
        motion="flow_reversal",
        geometry="sparse_23",
        speed_km_s=1.0,
        direction_deg=0.0,
        integration_time_sec=120,
        seed=seed,
    )


def matrix_cases(scope: str) -> list[DataCase]:
    """Expand one declarative ordinary-motion benchmark matrix."""

    settings = BENCHMARK_CONFIG["scopes"][scope]
    return [
        ordinary_case(geometry, float(speed), int(integration), int(seed))
        for geometry in settings["geometries"]
        for speed in settings["speeds_km_s"]
        for integration in settings["integration_times_sec"]
        for seed in settings["seeds"]
    ]


def build_cases(scope: str) -> list[DataCase]:
    if scope == "smoke":
        return [ordinary_case("sparse_23", 0.36, 60, 0), shear_case(0)]

    if scope == "pilot":
        settings = BENCHMARK_CONFIG["scopes"][scope]
        return matrix_cases(scope) + [
            shear_case(int(seed)) for seed in settings["flow_reversal_seeds"]
        ]

    if scope == "overnight_125":
        return matrix_cases(scope)

    core_settings = BENCHMARK_CONFIG["scopes"]["core"]
    core = matrix_cases("core")
    core.extend(shear_case(int(seed)) for seed in core_settings["flow_reversal_seeds"])
    if scope == "core":
        return core

    direction = BENCHMARK_CONFIG["full_direction_extension"]
    direction_cases = [
        ordinary_case(
            str(direction["geometry"]),
            float(speed),
            int(direction["integration_time_sec"]),
            int(seed),
            float(direction_deg),
        )
        for speed in direction["speeds_km_s"]
        for direction_deg in direction["directions_deg"]
        for seed in direction["seeds"]
    ]
    unique = {case.case_id: case for case in core + direction_cases}
    return list(unique.values())


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def generate_case(case: DataCase, root: Path, python: str, dry_run: bool) -> Path:
    data_dir = root / "data" / case.case_id
    expected = [
        data_dir / "synthetic_observations.csv",
        data_dir / "synthetic_config.json",
        data_dir / "synthetic_sample_geometry.csv",
    ]
    if all(path.exists() for path in expected):
        print(f"[generate] complete; skipping {case.case_id}")
        return data_dir
    if data_dir.exists() and any(data_dir.iterdir()):
        raise RuntimeError(f"Incomplete data directory exists: {data_dir}")

    command = [
        python,
        str(SOURCE_DIR / "synthetic_plasma.py"),
        "--output_dir",
        str(data_dir),
        "--duration_sec",
        str(case.duration_sec),
        "--n_times",
        str(case.n_times),
        "--motion",
        case.motion,
        "--sample_geometry",
        case.geometry,
        "--speed_km_s",
        str(case.speed_km_s),
        "--direction_deg",
        str(case.direction_deg),
        "--integration_time_sec",
        str(case.integration_time_sec),
        "--integration_samples",
        str(case.integration_samples),
        "--seed",
        str(case.seed),
    ]
    run_command(command, dry_run)
    return data_dir


def train_case(
    case: DataCase,
    regularization: str,
    root: Path,
    python: str,
    num_steps: int,
    cpu: bool,
    dry_run: bool,
) -> Path:
    data_dir = root / "data" / case.case_id
    run_dir = root / "runs" / case.case_id / regularization
    if (run_dir / "model_final.pt").exists() and (run_dir / "history.csv").exists():
        print(f"[train] complete; skipping {case.case_id}/{regularization}")
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"[train] incomplete; restarting {case.case_id}/{regularization}")

    command = [
        python,
        str(SOURCE_DIR / "synthetic_train_3d.py"),
        "--synthetic_csv",
        str(data_dir / "synthetic_observations.csv"),
        "--window_start_index",
        "0",
        "--window_size_records",
        str(case.n_times),
        "--num_steps",
        str(num_steps),
        "--epsilon_data",
        "1e-6",
        "--lambda_warmup_steps",
        "500",
        "--summary_every",
        "500",
        "--component_grad_every",
        "500",
        "--num_collocation",
        "16384",
        "--num_diagnostic_collocation",
        "16384",
        "--collocation_grid_nx",
        "100",
        "--collocation_grid_ny",
        "100",
        "--seed",
        str(case.seed),
        "--output_dir",
        str(run_dir),
    ]
    command.extend(REGULARIZATIONS[regularization])
    if cpu:
        command.append("--cpu")
    run_command(command, dry_run)
    return run_dir


def analyze_case(case: DataCase, regularization: str, root: Path, dry_run: bool) -> None:
    run_dir = root / "runs" / case.case_id / regularization
    if dry_run:
        print(f"analyze {case.case_id}/{regularization}")
        return
    if not (run_dir / "model_final.pt").exists():
        raise FileNotFoundError(f"Missing trained model: {run_dir / 'model_final.pt'}")

    import synthetic_analyze_reconstruction as analysis

    settings = analysis.ReconstructionAnalysisConfig(
        run_dir=run_dir,
        synthetic_csv=root / "data" / case.case_id / "synthetic_observations.csv",
        synthetic_config=root / "data" / case.case_id / "synthetic_config.json",
        time_selection="first_middle_last",
        compute_gradient_errors=True,
        save_dense_csv=True,
    )
    analysis.analyze_reconstruction(settings)


def write_manifest(cases: list[DataCase], root: Path, num_steps: int) -> None:
    rows = []
    for case in cases:
        for regularization in REGULARIZATIONS:
            row = asdict(case)
            row.update(
                {
                    "n_times": case.n_times,
                    "integration_samples": case.integration_samples,
                    "regularization": regularization,
                    "num_steps": num_steps,
                    "data_dir": str(root / "data" / case.case_id),
                    "run_dir": str(root / "runs" / case.case_id / regularization),
                }
            )
            rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "benchmark_manifest.csv", index=False)
    with open(root / "benchmark_config.json", "w") as f:
        json.dump({"cases": [asdict(case) for case in cases], "regularizations": REGULARIZATIONS}, f, indent=2)


def curvature_health(cases: list[DataCase], root: Path) -> pd.DataFrame:
    rows = []
    for case in cases:
        for regularization in REGULARIZATIONS:
            history_path = root / "runs" / case.case_id / regularization / "history.csv"
            if not history_path.exists():
                continue
            history = pd.read_csv(history_path)
            tail = history.iloc[max(0, int(0.75 * len(history))):]
            row: dict[str, object] = {**asdict(case), "regularization": regularization}
            for col in (
                "fxx_rms", "fxy_rms", "fyy_rms", "ftt_rms",
                "fxx_frac_near_zero", "fxy_frac_near_zero",
                "fyy_frac_near_zero", "ftt_frac_near_zero",
            ):
                values = pd.to_numeric(tail.get(col, pd.Series(dtype=float)), errors="coerce")
                row[f"tail_median_{col}"] = float(values.median()) if values.notna().any() else np.nan

            spatial_rms = [row[f"tail_median_{col}"] for col in ("fxx_rms", "fxy_rms", "fyy_rms")]
            spatial_zero = [row[f"tail_median_{col}"] for col in (
                "fxx_frac_near_zero", "fxy_frac_near_zero", "fyy_frac_near_zero"
            )]
            row["spatial_curvature_collapsed"] = bool(
                np.nanmax(spatial_rms) <= 1.0e-12 or np.nanmin(spatial_zero) >= 0.99
            )
            row["temporal_curvature_collapsed"] = bool(
                row["tail_median_ftt_rms"] <= 1.0e-12
                or row["tail_median_ftt_frac_near_zero"] >= 0.99
            )
            rows.append(row)
    report = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    report.to_csv(root / "curvature_health.csv", index=False)
    return report


def collect_metrics(cases: list[DataCase], root: Path) -> None:
    rows = []
    for case in cases:
        for regularization in REGULARIZATIONS:
            path = root / "runs" / case.case_id / regularization / "error_analysis" / "error_summary_mean.csv"
            if not path.exists():
                continue
            row = pd.read_csv(path).iloc[0].to_dict()
            row.update({**asdict(case), "regularization": regularization})
            rows.append(row)
    pd.DataFrame(rows).to_csv(root / "benchmark_metrics.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["smoke", "pilot", "overnight_125", "core", "full"],
        default="pilot",
    )
    parser.add_argument("--stage", choices=["manifest", "generate", "train", "analyze", "check", "all"], default="generate")
    parser.add_argument("--output_root", type=Path, default=Path("outputs/velocity_integration_benchmark"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num_steps", type=int, default=15000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.output_root.resolve()
    cases = build_cases(args.scope)
    write_manifest(cases, root, args.num_steps)
    print(f"Benchmark scope={args.scope}: {len(cases)} data cases, {len(cases) * len(REGULARIZATIONS)} training runs")

    if args.stage in ("generate", "all"):
        for case in cases:
            generate_case(case, root, args.python, args.dry_run)
    if args.stage in ("train", "all"):
        for case in cases:
            if not (root / "data" / case.case_id / "synthetic_observations.csv").exists():
                generate_case(case, root, args.python, args.dry_run)
            for regularization in REGULARIZATIONS:
                train_case(case, regularization, root, args.python, args.num_steps, args.cpu, args.dry_run)
    if args.stage in ("analyze", "all"):
        for case in cases:
            for regularization in REGULARIZATIONS:
                analyze_case(case, regularization, root, args.dry_run)
        if not args.dry_run:
            collect_metrics(cases, root)
    if args.stage in ("check", "all") and not args.dry_run:
        report = curvature_health(cases, root)
        if len(report):
            print(report[["case_id", "regularization", "spatial_curvature_collapsed", "temporal_curvature_collapsed"]])


if __name__ == "__main__":
    main()
