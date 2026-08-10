"""Manifest-driven, resumable experiment runner with no-side-effect dry runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .checkpoint import atomic_json_save
from .config import FourDConfig, apply_explicit_overrides, config_from_mapping
from .evaluation import calibrate_checkpoint, evaluate_checkpoint
from .pfisr import PFISRReadConfig, inspect_pfisr_hdf5, read_pfisr_4d
from .splits import GroupSplit4D, make_group_split, prepare_training_problem
from .synthetic import (
    MovingFeature4D,
    SyntheticFieldConfig,
    SyntheticObservationConfig,
    generate_synthetic_case,
)
from .synthetic_evaluation import evaluate_independent_synthetic_truth
from .training import train_4d


STAGES = ("manifest", "generate", "train", "calibrate", "evaluate", "summarize", "check", "all")
CASE_FIELDS = {"name", "question", "data", "split", "training", "conformal", "verification", "repetition_seeds", "synthetic_truth"}
SPLIT_FIELDS = {
    "unit",
    "strategy",
    "validation_group_count",
    "calibration_group_count",
    "test_group_count",
    "seed",
}
CONFORMAL_FIELDS = {"alpha", "bootstrap_repetitions"}
VERIFICATION_FIELDS = {"interrupt_after_steps", "resume_immediately"}
SYNTHETIC_TRUTH_FIELDS = {"count", "seed_offset"}


def _strict(mapping: Mapping[str, Any], allowed: set[str], name: str, required: set[str] | None = None) -> None:
    unknown = sorted(set(mapping) - allowed)
    missing = sorted((required or set()) - set(mapping))
    if unknown or missing:
        raise ValueError(f"Invalid {name}; unknown={unknown}, missing={missing}")


def _dataclass_from_mapping(cls: type, values: Mapping[str, Any], name: str):
    allowed = {field.name for field in fields(cls)}
    _strict(values, allowed, name)
    return cls(**values)


def load_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _strict(manifest, {"schema_version", "purpose", "output_root", "defaults", "cases"}, "manifest", {"schema_version", "purpose", "output_root", "cases"})
    if manifest["schema_version"] != 1 or not isinstance(manifest["cases"], list) or not manifest["cases"]:
        raise ValueError("Manifest schema_version must be 1 and cases must be non-empty.")
    if manifest["purpose"] not in {"smoke", "pilot", "long"}:
        raise ValueError("Manifest purpose must be smoke, pilot, or long.")
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("Manifest defaults must be an object.")

    def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(base))
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    merged_cases = [merge(defaults, case) for case in manifest["cases"]]
    for index, case in enumerate(merged_cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"Case {index} is not an object.")
        _strict(
            case,
            CASE_FIELDS,
            f"case {index}",
            CASE_FIELDS - {"verification", "repetition_seeds", "synthetic_truth"},
        )
        if not case["name"] or not case["question"]:
            raise ValueError(f"Case {index} requires non-empty name and question.")
        _strict(case["split"], SPLIT_FIELDS, f"case {index} split", SPLIT_FIELDS)
        _strict(case["conformal"], CONFORMAL_FIELDS, f"case {index} conformal", CONFORMAL_FIELDS)
        _strict(case.get("verification", {}), VERIFICATION_FIELDS, f"case {index} verification")
        _strict(case.get("synthetic_truth", {}), SYNTHETIC_TRUTH_FIELDS, f"case {index} synthetic truth")
        config_from_mapping(case["training"])
        _validate_data(case["data"])
    expanded = []
    for case in merged_cases:
        seeds = case.get("repetition_seeds")
        if seeds is None:
            expanded.append(dict(case))
            continue
        if not isinstance(seeds, list) or not seeds or any(
            not isinstance(seed, int) or seed < 0 for seed in seeds
        ):
            raise ValueError("repetition_seeds must be a non-empty list of non-negative integers.")
        if len(seeds) != len(set(seeds)):
            raise ValueError("repetition_seeds cannot contain duplicates.")
        for seed in seeds:
            repeated = json.loads(json.dumps(case))
            repeated.pop("repetition_seeds", None)
            repeated["name"] = f"{case['name']} seed {seed}"
            repeated["split"]["seed"] = seed
            repeated["training"].setdefault("optimization", {})["seed"] = seed
            repeated["training"].setdefault("collocation", {})["seed"] = seed
            if repeated["data"]["type"] == "synthetic":
                repeated["data"]["observation"]["seed"] = seed
            expanded.append(repeated)
    manifest = dict(manifest)
    manifest.pop("defaults", None)
    manifest["_source_path"] = str(Path(path).resolve())
    manifest["cases"] = expanded
    ids = [case_identifier(case) for case in manifest["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest expands to duplicate deterministic case identifiers.")
    return manifest


def _validate_data(data: Mapping[str, Any]) -> None:
    if data.get("type") == "synthetic":
        _strict(data, {"type", "field", "observation"}, "synthetic data", {"type", "field", "observation"})
        field_values = dict(data["field"])
        feature_values = field_values.pop("feature", {})
        feature = _dataclass_from_mapping(MovingFeature4D, feature_values, "synthetic feature")
        _dataclass_from_mapping(SyntheticFieldConfig, {**field_values, "feature": feature}, "synthetic field").validate()
        _dataclass_from_mapping(SyntheticObservationConfig, data["observation"], "synthetic observation").validate()
    elif data.get("type") == "pfisr":
        _strict(data, {"type", "path", "reader"}, "PFISR data", {"type", "path", "reader"})
        _dataclass_from_mapping(PFISRReadConfig, data["reader"], "PFISR reader").validate()
    else:
        raise ValueError(f"Unknown case data type: {data.get('type')}")


def case_identifier(case: Mapping[str, Any]) -> str:
    semantic = {key: case[key] for key in ("data", "split", "training", "conformal")}
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", str(case["name"]).lower()).strip("-")[:60]
    return f"{slug}-{digest}"


def _build_data(case: Mapping[str, Any]):
    data = case["data"]
    if data["type"] == "synthetic":
        field_values = dict(data["field"])
        feature = _dataclass_from_mapping(
            MovingFeature4D, field_values.pop("feature", {}), "synthetic feature"
        )
        field_config = _dataclass_from_mapping(
            SyntheticFieldConfig, {**field_values, "feature": feature}, "synthetic field"
        )
        observation_config = _dataclass_from_mapping(
            SyntheticObservationConfig, data["observation"], "synthetic observation"
        )
        generated = generate_synthetic_case(field_config, observation_config)
        metadata = {
            "type": "synthetic",
            "field": {**asdict(field_config)},
            "observation": asdict(observation_config),
            "observations": generated.bundle.size,
            "beams": len(np.unique(generated.bundle.beam_ids)),
            "time_blocks": len(np.unique(generated.bundle.time_ids)),
            "integration_duration_unique_sec": np.unique(generated.integration_duration_sec).tolist(),
            "target_semantics": "integration-product log10(mean linear Ne)" if observation_config.mode == "integration_averaged" else "instantaneous log10 Ne",
            "midpoint_truth_saved_in_memory_for_distinct_smearing_analysis": True,
        }
        return generated.bundle, metadata
    reader = _dataclass_from_mapping(PFISRReadConfig, data["reader"], "PFISR reader")
    generated = read_pfisr_4d(Path(data["path"]), reader)
    finite_relative = generated.relative_uncertainty[np.isfinite(generated.relative_uncertainty)]
    selected_records = np.unique(generated.record_indices)
    record_timing = [
        {
            "record_index": int(record),
            "start_unix": float(generated.metadata.unix_start[record]),
            "end_unix": float(generated.metadata.unix_end[record]),
            "midpoint_unix": float(generated.metadata.unix_mid[record]),
            "duration_sec": float(generated.metadata.integration_duration_sec[record]),
        }
        for record in selected_records
    ]
    metadata = {
        "type": "pfisr",
        "file": generated.metadata.state_dict(),
        "reader": asdict(reader),
        "observations": generated.bundle.size,
        "beams": len(np.unique(generated.bundle.beam_ids)),
        "time_blocks": len(np.unique(generated.bundle.time_ids)),
        "selected_record_indices": selected_records.tolist(),
        "selected_record_timing": record_timing,
        "selected_unix_start": float(generated.unix_start.min()),
        "selected_unix_end": float(generated.unix_end.max()),
        "integration_duration_unique_sec": np.unique(generated.integration_duration_sec).tolist(),
        "exclusions": generated.exclusions,
        "uncertainty": {
            "finite_positive_count": int(np.sum(np.isfinite(generated.uncertainty_ne_m3) & (generated.uncertainty_ne_m3 > 0))),
            "nonfinite_count": int(np.sum(~np.isfinite(generated.uncertainty_ne_m3))),
            "relative_finite_min": float(finite_relative.min()) if len(finite_relative) else None,
            "relative_finite_max": float(finite_relative.max()) if len(finite_relative) else None,
            "training_weighting": "not applied; unweighted MSE comparator",
        },
    }
    return generated.bundle, metadata


def _build_split(bundle, case: Mapping[str, Any]) -> GroupSplit4D:
    return make_group_split(bundle, **case["split"])


def _case_directory(manifest: Mapping[str, Any], case: Mapping[str, Any], attempt_id: str) -> Path:
    base = Path(manifest["output_root"]).expanduser()
    identifier = case_identifier(case)
    return base / (identifier if attempt_id == "main" else f"{identifier}--{attempt_id}")


def _completion_valid(path: Path, required_files: Iterable[Path] = ()) -> bool:
    if not path.is_file():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("schema_version") == 1 and state.get("status") == "complete" and all(file.is_file() for file in required_files)


def _plan_case(manifest: Mapping[str, Any], case: Mapping[str, Any], attempt_id: str) -> dict[str, Any]:
    bundle, data_metadata = _build_data(case)
    split = _build_split(bundle, case)
    config = config_from_mapping(case["training"])
    case_directory = _case_directory(manifest, case, attempt_id)
    return {
        "case_id": case_identifier(case),
        "name": case["name"],
        "question": case["question"],
        "case_directory": str(case_directory),
        "data": data_metadata,
        "split_counts": split.state_dict()["counts"],
        "collocation": {
            "mode": config.collocation.mode,
            "pool_size": config.collocation.pool_size,
            "batch_size": config.collocation.batch_size,
            "derivative_microbatch_size": config.collocation.derivative_microbatch_size,
            "normalized_domain_lower": list(config.collocation.domain_lower),
            "normalized_domain_upper": list(config.collocation.domain_upper),
        },
        "data_batch_size": config.optimization.data_batch_size,
        "optimization_seed": config.optimization.seed,
        "inference_chunk_size": config.runtime.inference_chunk_size,
        "diagnostic_probe_size": config.runtime.diagnostic_probe_size,
        "requested_device": config.runtime.device,
        "precision": config.runtime.precision,
        "amp": config.runtime.amp,
        "planned_commands": {
            "train": f"python -m inr_isr_4d.runner {manifest['_source_path']} train --case-id {case_identifier(case)}",
            "resume": f"python -m inr_isr_4d.runner {manifest['_source_path']} train --case-id {case_identifier(case)} --resume",
        },
        "planned_outputs": {
            "checkpoint": str(case_directory / "train" / "checkpoint.pt"),
            "calibration": str(case_directory / "uq" / "calibration.json"),
            "evaluation": str(case_directory / "evaluation" / "evaluation_summary.json"),
        },
    }


def dry_run(manifest: Mapping[str, Any], cases: list[Mapping[str, Any]], attempt_id: str) -> list[dict[str, Any]]:
    return [_plan_case(manifest, case, attempt_id) for case in cases]


def _write_generation(case_directory: Path, case: Mapping[str, Any], metadata: dict[str, Any], split: GroupSplit4D) -> None:
    case_directory.mkdir(parents=True, exist_ok=False)
    atomic_json_save(dict(case), case_directory / "case.json")
    atomic_json_save(metadata, case_directory / "data_summary.json")
    atomic_json_save(split.state_dict(), case_directory / "splits.json")
    atomic_json_save(
        {"schema_version": 1, "status": "complete", "case_id": case_identifier(case)},
        case_directory / "GENERATED.json",
    )


def _failure(case_directory: Path, stage: str, error: BaseException) -> None:
    case_directory.mkdir(parents=True, exist_ok=True)
    atomic_json_save(
        {
            "schema_version": 1,
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
        case_directory / f"FAILED_{stage}.json",
    )


def run_case_stage(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    stage: str,
    *,
    attempt_id: str,
    resume: bool,
) -> dict[str, Any]:
    directory = _case_directory(manifest, case, attempt_id)
    generated_marker = directory / "GENERATED.json"
    bundle, metadata = _build_data(case)
    split = _build_split(bundle, case)
    problem = prepare_training_problem(bundle, split)
    config = config_from_mapping(case["training"])

    if stage == "generate":
        if _completion_valid(generated_marker):
            return {"stage": stage, "status": "skipped_complete"}
        if directory.exists():
            raise FileExistsError(f"Case directory collision: {directory}")
        _write_generation(directory, case, metadata, split)
        return {"stage": stage, "status": "complete"}
    if not _completion_valid(generated_marker, [directory / "case.json", directory / "splits.json"]):
        raise RuntimeError("Generate stage is incomplete; run generate first.")

    if stage == "train":
        completed = directory / "train" / "COMPLETED.json"
        if _completion_valid(completed, [directory / "train" / "checkpoint.pt"]):
            return {"stage": stage, "status": "skipped_complete"}
        train_directory = directory / "train"
        verification = case.get("verification", {})
        interrupt = verification.get("interrupt_after_steps")
        resume_immediately = bool(verification.get("resume_immediately", False))
        has_checkpoint = (train_directory / "checkpoint.pt").is_file()
        result = train_4d(
            problem,
            config,
            train_directory,
            resume=resume or has_checkpoint,
            stop_after_steps=None if has_checkpoint else interrupt,
        )
        if not result.complete and resume_immediately:
            result = train_4d(problem, config, train_directory, resume=True)
        return {"stage": stage, "status": "complete" if result.complete else "interrupted", "steps": result.steps_completed}

    checkpoint = directory / "train" / "checkpoint.pt"
    if not _completion_valid(directory / "train" / "COMPLETED.json", [checkpoint]):
        raise RuntimeError("Train stage is incomplete.")
    if stage == "calibrate":
        path = directory / "uq" / "calibration.json"
        if path.is_file():
            return {"stage": stage, "status": "skipped_complete"}
        calibrate_checkpoint(
            bundle=bundle,
            split=split,
            coordinate_scaler=problem.coordinate_scaler,
            target_scaler=problem.target_scaler,
            checkpoint_path=checkpoint,
            calibration_path=path,
            alpha=float(case["conformal"]["alpha"]),
        )
        return {"stage": stage, "status": "complete"}

    calibration = directory / "uq" / "calibration.json"
    if not calibration.is_file():
        raise RuntimeError("Calibrate stage is incomplete.")
    if stage == "evaluate":
        completion = directory / "evaluation" / "COMPLETED.json"
        required_evaluation = [
            directory / "evaluation" / "predictions.npz",
            directory / "evaluation" / "metrics.csv",
            directory / "evaluation" / "stratified_intervals.csv",
        ]
        if case["data"]["type"] == "synthetic":
            required_evaluation.extend(
                [
                    directory / "synthetic_truth" / "summary.json",
                    directory / "synthetic_truth" / "predictions.npz",
                ]
            )
        if _completion_valid(completion, required_evaluation):
            return {"stage": stage, "status": "skipped_complete"}
        evaluate_checkpoint(
            bundle=bundle,
            split=split,
            coordinate_scaler=problem.coordinate_scaler,
            target_scaler=problem.target_scaler,
            checkpoint_path=checkpoint,
            calibration_path=calibration,
            output_directory=directory / "evaluation",
            bootstrap_repetitions=int(case["conformal"]["bootstrap_repetitions"]),
        )
        if case["data"]["type"] == "synthetic":
            field_values = dict(case["data"]["field"])
            feature = _dataclass_from_mapping(
                MovingFeature4D, field_values.pop("feature", {}), "synthetic feature"
            )
            field_config = _dataclass_from_mapping(
                SyntheticFieldConfig,
                {**field_values, "feature": feature},
                "synthetic field",
            )
            observation_config = _dataclass_from_mapping(
                SyntheticObservationConfig,
                case["data"]["observation"],
                "synthetic observation",
            )
            truth_settings = case.get("synthetic_truth", {})
            evaluate_independent_synthetic_truth(
                bundle=bundle,
                field_config=field_config,
                observation_config=observation_config,
                coordinate_scaler=problem.coordinate_scaler,
                target_scaler=problem.target_scaler,
                checkpoint_path=checkpoint,
                output_directory=directory / "synthetic_truth",
                truth_count=int(truth_settings.get("count", 1024)),
                truth_seed=int(case["split"]["seed"]) + int(truth_settings.get("seed_offset", 100000)),
            )
        return {"stage": stage, "status": "complete"}

    if stage == "summarize":
        evaluation = directory / "evaluation" / "evaluation_summary.json"
        if not _completion_valid(directory / "evaluation" / "COMPLETED.json", [evaluation]):
            raise RuntimeError("Evaluation stage is incomplete.")
        summary_path = directory / "SUMMARY.json"
        if summary_path.is_file():
            return {"stage": stage, "status": "skipped_complete"}
        evaluation_state = json.loads(evaluation.read_text(encoding="utf-8"))
        synthetic_truth_path = directory / "synthetic_truth" / "summary.json"
        synthetic_truth = (
            json.loads(synthetic_truth_path.read_text(encoding="utf-8"))
            if synthetic_truth_path.is_file()
            else None
        )
        atomic_json_save(
            {
                "schema_version": 1,
                "status": "complete",
                "case_id": case_identifier(case),
                "question": case["question"],
                "data": metadata,
                "point_metrics_log10_density": evaluation_state["point_metrics_log10_density"],
                "marginal_interval_metrics": evaluation_state["marginal_interval_metrics"],
                "independent_synthetic_truth": synthetic_truth,
                "claim_boundary": "Executable smoke/pilot capability is not evidence of scientific superiority or implemented plasma physics.",
            },
            summary_path,
        )
        return {"stage": stage, "status": "complete"}

    if stage == "check":
        required = [
            generated_marker,
            directory / "train" / "COMPLETED.json",
            calibration,
            directory / "evaluation" / "COMPLETED.json",
            directory / "SUMMARY.json",
        ]
        if case["data"]["type"] == "synthetic":
            required.extend(
                [
                    directory / "synthetic_truth" / "summary.json",
                    directory / "synthetic_truth" / "predictions.npz",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Case completion validation failed; missing={missing}")
        return {"stage": stage, "status": "complete", "validated": [str(path) for path in required]}
    raise ValueError(f"Unsupported executable stage: {stage}")


def _selected_cases(manifest: Mapping[str, Any], selector: str | None) -> list[Mapping[str, Any]]:
    cases = manifest["cases"]
    if selector is None:
        return cases
    selected = [case for case in cases if case_identifier(case) == selector or case["name"] == selector]
    if not selected:
        raise ValueError(f"No manifest case matches {selector}.")
    return selected


def execute(
    manifest: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    stage: str,
    *,
    attempt_id: str,
    resume: bool,
) -> list[dict[str, Any]]:
    sequence = (
        ("generate", "train", "calibrate", "evaluate", "summarize", "check")
        if stage == "all"
        else (stage,)
    )
    results = []
    for case in cases:
        directory = _case_directory(manifest, case, attempt_id)
        for item in sequence:
            try:
                outcome = run_case_stage(
                    manifest, case, item, attempt_id=attempt_id, resume=resume
                )
                results.append({"case_id": case_identifier(case), **outcome})
            except BaseException as error:
                _failure(directory, item, error)
                raise
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--case-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--attempt-id", default="main")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.FIELD=JSON_VALUE",
        help="Explicit strict 4D configuration override; may be repeated.",
    )
    return parser


def _explicit_cli_overrides(items: list[str], seed: int | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if seed is not None:
        overrides["optimization.seed"] = seed
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value without '=': {item}")
        name, encoded = item.split("=", 1)
        if name in overrides:
            raise ValueError(f"Configuration override supplied more than once: {name}")
        try:
            overrides[name] = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON value in --set {item}") from error
    return overrides


def _override_cases(
    cases: list[Mapping[str, Any]], overrides: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if not overrides:
        return cases
    result = []
    for case in cases:
        updated = json.loads(json.dumps(case))
        config = apply_explicit_overrides(
            config_from_mapping(updated["training"]), overrides
        )
        updated["training"] = config.to_dict()
        result.append(updated)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.restart and args.attempt_id == "main":
        raise ValueError("--restart requires a new explicit --attempt-id to preserve historical runs.")
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.attempt_id):
        raise ValueError("attempt-id contains unsupported characters.")
    manifest = load_manifest(args.manifest)
    cases = _selected_cases(manifest, args.case_id)
    seed = args.seed if hasattr(args, "seed") else None
    cases = _override_cases(cases, _explicit_cli_overrides(args.set, seed))
    if args.stage == "manifest":
        result: Any = [
            {"case_id": case_identifier(case), "name": case["name"], "question": case["question"]}
            for case in cases
        ]
    elif args.dry_run:
        result = dry_run(manifest, cases, args.attempt_id)
    else:
        result = execute(
            manifest,
            cases,
            args.stage,
            attempt_id=args.attempt_id,
            resume=args.resume,
        )
    json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
