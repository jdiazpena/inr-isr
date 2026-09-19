"""Explicit four-route INR training and saved-model query front door.

Source, named coordinates, preprocessing and *native scientific recipe* are
independent declarations. This module selects existing owners; it does not
define a new loss, refit checkpoint scalers, or run conformal calibration.

Examples:
  python -m inr_isr.workflow train --config case.json --output fresh-run
  python -m inr_isr.workflow query --config case.json --checkpoint old.pt --output fresh-query
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

from .data import InputSpec, PreprocessingSpec, prepare_input


_XYT_RECIPES = {
    "active_xyt_synthetic": ("synthetic", "synthetic_train_3d"),
    "active_xyt_radar": ("radar", "train_radar_3d_window_reference_reg_diagnostic"),
}
_RADAR_SELECTION = {
    "h0_km", "half_width_km", "time_start_utc", "time_end_utc",
    "record_stride", "window_start_index", "window_size_records",
}
_SYNTHETIC_SELECTION = {
    "target_col", "window_start_index", "window_size_records",
}
_FROZEN_MARKERS = (
    "REFERENCE_MANIFEST.json", "R2_FINAL_MANIFEST.json",
    "R3_FINAL_MANIFEST.json", "R4_FINAL_MANIFEST.json",
    "R5_FINAL_MANIFEST.json", "R6_FINAL_MANIFEST.json",
)


@dataclass(frozen=True)
class WorkflowSpec:
    """Validated declarations; native arguments/config are resolved separately."""

    input: InputSpec
    preprocessing: PreprocessingSpec
    recipe_name: str
    recipe_values: Mapping[str, object]
    query: Mapping[str, object] | None
    raw: Mapping[str, object]


def _only(mapping: Mapping[str, object], allowed: set[str], context: str) -> None:
    """Make typos fail instead of silently choosing a default or other recipe."""
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {context} fields: {sorted(unknown)}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    """Require the JSON object expected at each structural boundary."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object.")
    return value


def load_spec(path: Path) -> WorkflowSpec:
    """Read one JSON config; resolve file input relative to the config location."""
    path = Path(path).resolve()
    raw = _mapping(json.loads(path.read_text()), "workflow")
    _only(raw, {"input", "preprocessing", "training_recipe", "query"}, "workflow")
    for required in ("input", "preprocessing", "training_recipe"):
        if required not in raw:
            raise ValueError(f"Workflow requires {required}.")
    input_values = dict(_mapping(raw["input"], "input"))
    if input_values.get("path") is not None:
        source_path = Path(input_values["path"])
        # Old native run_config/checkpoint metadata preserves the literal absolute
        # path supplied by its caller. Only relative paths need a config-relative
        # resolution; canonicalizing /home versus /mnt would alter that record.
        input_values["path"] = (str(source_path) if source_path.is_absolute()
                                else str((path.parent / source_path).resolve()))
    inputs = InputSpec(**input_values)
    preprocessing = PreprocessingSpec(
        **dict(_mapping(raw["preprocessing"], "preprocessing"))
    )
    expected_policy = "selected_window" if inputs.coordinates == "xyt" else "training_split"
    if preprocessing.policy != expected_policy:
        raise ValueError(f"{inputs.coordinates} requires {expected_policy} preprocessing.")
    recipe = _mapping(raw["training_recipe"], "training_recipe")
    name = recipe.get("name")
    if name in _XYT_RECIPES:
        _only(recipe, {"name", "parameters"}, "xyt training_recipe")
        source, _ = _XYT_RECIPES[name]
        if inputs.coordinates != "xyt" or inputs.source != source:
            raise ValueError(f"{name} requires {source} xyt input.")
        if not preprocessing.normalize_coords or not preprocessing.normalize_values:
            raise ValueError("Active xyt trainers require their native normalization.")
        allowed_selection = _RADAR_SELECTION if source == "radar" else _SYNTHETIC_SELECTION
        _only(inputs.selection, allowed_selection, "active xyt input selection")
        values = _mapping(recipe.get("parameters", {}), "xyt parameters")
    elif name == "native_xyzt":
        _only(recipe, {"name", "config"}, "xyzt training_recipe")
        if inputs.coordinates != "xyzt":
            raise ValueError("native_xyzt requires xyzt input.")
        if "config" not in recipe:
            raise ValueError("native_xyzt requires an explicit native config mapping.")
        values = _mapping(recipe["config"], "xyzt config")
        from inr_isr_4d.config import config_from_mapping
        config_from_mapping(values)
    else:
        raise ValueError(f"Unknown native training recipe: {name!r}")
    if inputs.path is not None and not Path(inputs.path).is_file():
        raise FileNotFoundError(inputs.path)
    query = raw.get("query")
    if query is not None:
        query = _mapping(query, "query")
        kind = query.get("kind")
        if kind == "observations":
            _only(query, {"kind", "chunk_size", "device"}, "observations query")
        elif kind == "observed_grid":
            _only(query, {"kind", "nx", "ny", "chunk_size", "device"}, "grid query")
            for key in ("nx", "ny"):
                if type(query.get(key)) is not int or query[key] <= 0:
                    raise ValueError(f"query.{key} must be a positive integer.")
        else:
            raise ValueError(f"Unknown query kind: {kind!r}")
        requested_chunk = query.get("chunk_size")
        if requested_chunk != "full" or kind != "observations":
            if type(requested_chunk) is not int or requested_chunk <= 0:
                raise ValueError("query.chunk_size must be positive, or full for observations.")
        if "device" in query:
            # Validate syntax here; availability is checked at execution time.
            requested = query["device"]
            if not isinstance(requested, str) or not (
                requested in {"auto", "cuda", "cpu"}
                or (requested.startswith("cuda:") and requested[5:].isdigit())
            ):
                raise ValueError(f"Invalid query.device: {requested!r}")
    return WorkflowSpec(inputs, preprocessing, str(name), values, query, raw)


def _fresh_output(path: Path) -> Path:
    """Reject occupied and sealed ancestors before an old trainer can write."""
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    for parent in output.parents:
        if any((parent / marker).exists() for marker in _FROZEN_MARKERS):
            raise FileExistsError(f"Output is inside a frozen evidence packet: {parent}")
    return output


def _native_value(action: argparse.Action, value: object) -> object:
    """Check JSON override types before assigning an existing argparse default."""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if type(value) is not bool:
            raise ValueError(f"{action.dest} must be boolean.")
    elif action.type is int:
        if type(value) is not int:
            raise ValueError(f"{action.dest} must be an integer.")
    elif action.type is float:
        if type(value) not in (int, float):
            raise ValueError(f"{action.dest} must be numeric.")
        value = float(value)
    elif action.type is str:
        if not isinstance(value, str):
            raise ValueError(f"{action.dest} must be a string.")
    if action.choices is not None and value not in action.choices:
        raise ValueError(f"Invalid {action.dest}: {value!r}")
    return value


def _xyt_arguments(spec: WorkflowSpec, output: Path):
    """Start with the active parser's own defaults, then add one source selector."""
    _, module_name = _XYT_RECIPES[spec.recipe_name]
    module = importlib.import_module(module_name)
    parser = module.build_parser()
    arguments = parser.parse_args([])
    actions = {action.dest: action for action in parser._actions}
    source_fields = _RADAR_SELECTION | _SYNTHETIC_SELECTION | {
        "synthetic_csv", "h5_path", "output_dir",
    }
    _only(spec.recipe_values, set(actions) - source_fields - {"help"},
          "xyt training parameters")
    for name, value in spec.recipe_values.items():
        setattr(arguments, name, _native_value(actions[name], value))
    for name, value in spec.input.selection.items():
        setattr(arguments, name, _native_value(actions[name], value))
    path_field = "h5_path" if spec.input.source == "radar" else "synthetic_csv"
    setattr(arguments, path_field, str(spec.input.path))
    arguments.output_dir = str(output / "training")
    return module, arguments


def train(spec: WorkflowSpec, output: Path) -> dict[str, object]:
    """Dispatch a declared native recipe without changing its update sequence."""
    output = _fresh_output(output)
    # Validate the old argument namespace before making a directory.
    if spec.recipe_name in _XYT_RECIPES:
        module, arguments = _xyt_arguments(spec, output)
        from .device import resolve_device
        resolve_device("cpu" if arguments.cpu else "cuda")
    else:
        from inr_isr_4d.config import config_from_mapping
        from .device import resolve_device
        configuration = config_from_mapping(spec.recipe_values)
        resolve_device(configuration.runtime.device)
    output.mkdir(parents=True, exist_ok=False)
    (output / "workflow_config.json").write_text(
        json.dumps(spec.raw, indent=2, allow_nan=False)
    )
    started = time.monotonic()
    try:
        if spec.recipe_name in _XYT_RECIPES:
            # train_window seeds before its dataset factory. Do not pre-read data.
            module.train(arguments)
            checkpoint = output / "training/model_final.pt"
        else:
            from inr_isr_4d.training import train_4d
            prepared = prepare_input(spec.input, spec.preprocessing)
            train_4d(prepared.problem, configuration, output / "train")
            checkpoint = output / "train/checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Native trainer did not create {checkpoint}")
    except BaseException as error:
        (output / "workflow_failure.json").write_text(json.dumps(
            dict(type=type(error).__name__, message=str(error),
                 executable=sys.executable, elapsed_seconds=time.monotonic() - started),
            indent=2, allow_nan=False))
        raise
    report = dict(recipe=spec.recipe_name, input_source=spec.input.source,
                  coordinates=spec.input.coordinates, checkpoint=str(checkpoint),
                  executable=sys.executable, elapsed_seconds=time.monotonic() - started)
    (output / "workflow_result.json").write_text(json.dumps(report, indent=2))
    return report


def _query_coordinates(spec: WorkflowSpec) -> np.ndarray:
    """Use configured physical observations or an x-fastest grid of their bounds."""
    assert spec.query is not None
    prepared = prepare_input(spec.input, spec.preprocessing)
    observed = prepared.physical_field.coordinates
    if spec.query["kind"] == "observations":
        return observed
    nx, ny = spec.query["nx"], spec.query["ny"]
    x, y = np.meshgrid(np.linspace(observed[:, 0].min(), observed[:, 0].max(), nx),
                       np.linspace(observed[:, 1].min(), observed[:, 1].max(), ny))
    if spec.input.coordinates == "xyt":
        return np.column_stack((x.ravel(), y.ravel(),
                                np.full(x.size, np.median(observed[:, 2]))))
    return np.column_stack((x.ravel(), y.ravel(),
                            np.full(x.size, np.median(observed[:, 2])),
                            np.full(x.size, np.median(observed[:, 3]))))


def _sha256(path: Path) -> str:
    """Bind a trusted local checkpoint without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query(spec: WorkflowSpec, checkpoint: Path, output: Path,
          *, device_policy: str | None = None) -> dict[str, object]:
    """Reload a named native checkpoint and predict physical log10 density."""
    if spec.query is None:
        raise ValueError("This workflow config has no query specification.")
    output = _fresh_output(output)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    from .device import resolve_device, device_record
    configured_policy = spec.query.get("device")
    if device_policy is not None and configured_policy is not None:
        def equivalent(value: str) -> str:
            return "cuda" if value == "auto" else value
        if equivalent(device_policy) != equivalent(str(configured_policy)):
            raise ValueError("CLI --device conflicts with query.device in the config.")
    requested_device = device_policy or configured_policy or "cuda"
    device = resolve_device(str(requested_device))
    started = time.monotonic()
    coordinates = _query_coordinates(spec)
    chunk = (len(coordinates) if spec.query["chunk_size"] == "full"
             else int(spec.query["chunk_size"]))
    observed_batch_device: str | None = None
    def observe_batch(_model, inputs):
        nonlocal observed_batch_device
        if observed_batch_device is None:
            observed_batch_device = str(inputs[0].device)
    if spec.input.coordinates == "xyt":
        from .checkpoints import torch_load_checkpoint, build_model_from_checkpoint
        from .transforms import normalize_coords_from_scalers
        from .prediction import predict_log10_on_coords
        state = torch_load_checkpoint(checkpoint, "cpu")
        if len(state["coord_scalers"]) != 3:
            raise ValueError("Named xyt query requires a three-coordinate checkpoint.")
        model = build_model_from_checkpoint(state, device)
        normalized = normalize_coords_from_scalers(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
            state["coord_scalers"])
        handle = model.register_forward_pre_hook(observe_batch)
        try:
            predictions = predict_log10_on_coords(
                model, normalized, state["target_scaler"], device, chunk)
        finally:
            handle.remove()
        dtype = next(model.parameters()).dtype
        amp = False
    else:
        from inr_isr_4d.data import AffineScaler
        from inr_isr_4d.evaluation import load_trained_model, predict_chunked
        model, config, state, device, dtype = load_trained_model(
            checkpoint, device_policy=str(requested_device))
        coordinate_scaler = AffineScaler(**{
            key: np.asarray(value)
            for key, value in state["preprocessing"]["coordinates"].items()
        })
        target_scaler = AffineScaler(**{
            key: np.asarray(value)
            for key, value in state["preprocessing"]["targets"].items()
        })
        handle = model.register_forward_pre_hook(observe_batch)
        try:
            predictions = predict_chunked(
                model, coordinates, coordinate_scaler, target_scaler,
                device=device, dtype=dtype, chunk_size=chunk)
        finally:
            handle.remove()
        amp = config.runtime.amp
    import torch
    actual_batch = torch.device(observed_batch_device) if observed_batch_device is not None else None
    if (actual_batch is None or actual_batch.type != device.type
            or (device.index is not None and actual_batch.index != device.index)):
        raise RuntimeError(f"Prediction batch ran on {observed_batch_device}, expected {device}.")
    execution = device_record(str(requested_device), device, model=model, dtype=dtype, amp=amp)
    execution["batch_device"] = observed_batch_device
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "queries.npz",
                        coordinates=coordinates, predictions=predictions)
    report = dict(recipe=spec.recipe_name, source=spec.input.source,
                  coordinates=spec.input.coordinates, query_kind=spec.query["kind"],
                  rows=len(coordinates), requested_chunk_size=spec.query["chunk_size"],
                  chunk_size=chunk,
                  checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
                  units=list(("km", "km", "s") if spec.input.coordinates == "xyt"
                             else ("km", "km", "km", "s")),
                  target="log10(Ne/m^-3)", executable=sys.executable,
                  execution_device=execution,
                  elapsed_seconds=time.monotonic() - started)
    (output / "query_result.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """Expose train/query as separate bounded commands with explicit outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "query"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "query":
            command.add_argument("--checkpoint", required=True, type=Path)
            command.add_argument("--device", type=str, default=None,
                                 help="Execution device: cuda (default), cpu, auto, or cuda:N")
    args = parser.parse_args(argv)
    spec = load_spec(args.config)
    result = (train(spec, args.output) if args.command == "train"
              else query(spec, args.checkpoint, args.output, device_policy=args.device))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
