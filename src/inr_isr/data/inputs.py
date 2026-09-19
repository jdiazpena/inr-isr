"""Configured data preparation with distinct native scientific policies.

Source selects a reader/generator, coordinate mode selects xyt/xyzt, preprocessing
selects a fit population. None selects a loss. Imports/readers are lazy so importing
the neutral representation does not open files, initialize networks or draw RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .observations import AXES, PhysicalField


@dataclass(frozen=True)
class InputSpec:
    """Explicit source/mode plus native selectors or analytic generator parameters.

Use a path for both xyt CSV/HDF5 and radar xyzt; synthetic xyzt instead needs
explicit field and observation mappings. Mappings are shallow copied, not deeply
frozen. Native parameter names/edge semantics are deliberately retained. Custom
legacy columns remain available through old imports, not mislabeled by this API.
"""
    source: str
    coordinates: str
    path: str | Path | None = None
    selection: Mapping[str, object] = field(default_factory=dict)
    field: Mapping[str, object] | None = None
    observation: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent recipes before any file/generator is accessed."""
        if self.source not in {"radar", "synthetic"} or self.coordinates not in AXES:
            raise ValueError("InputSpec requires source radar/synthetic and coordinates xyt/xyzt.")
        generated = self.source == "synthetic" and self.coordinates == "xyzt"
        if generated:
            if self.path is not None or self.selection or self.field is None or self.observation is None:
                raise ValueError("Synthetic xyzt needs field/observation only, not path/selection.")
        elif self.path is None or self.field is not None or self.observation is not None:
            raise ValueError("File inputs need path/selection, not field/observation.")
        if {"normalize_coords", "normalize_values"} & set(self.selection):
            raise ValueError("Normalization belongs in PreprocessingSpec, not source selection.")
        if self.coordinates == "xyt":
            if tuple(self.selection.get("coord_cols", AXES["xyt"])) != AXES["xyt"]:
                raise ValueError("Named xyt requires columns x_km,y_km,t_sec in that order.")
            if self.selection.get("target_col", "log10_Ne") != "log10_Ne":
                raise ValueError("Named physical input requires log10_Ne as its target.")
        object.__setattr__(self, "selection", dict(self.selection))
        if self.field is not None:
            object.__setattr__(self, "field", dict(self.field))
            object.__setattr__(self, "observation", dict(self.observation))


@dataclass(frozen=True)
class PreprocessingSpec:
    """Declare selected_window (xyt) or training_split (xyzt), never infer from source.

Normalization switches preserve the old xyt interface. xyzt always uses its
existing affine maps and needs an explicit native split mapping. Splits do not
implicitly request calibration; they only assign the existing row roles.
"""
    policy: str
    normalize_coords: bool = True
    normalize_values: bool = True
    split: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate policy structure while leaving numerical checks to native code."""
        if self.policy == "selected_window":
            if self.split is not None:
                raise ValueError("selected_window does not accept a group split.")
        elif self.policy == "training_split":
            if self.split is None or not self.normalize_coords or not self.normalize_values:
                raise ValueError("training_split needs explicit split and native normalization.")
            object.__setattr__(self, "split", dict(self.split))
        else:
            raise ValueError("Unknown preprocessing policy; use selected_window/training_split.")


@dataclass(frozen=True)
class PreparedInput:
    """Physical field plus full normalized float32 tensors and original native objects.

For xyt, tensors alias the old mutable dataset tensors; for xyzt, tensors own a
new conversion using the unchanged train-fitted scalers. train_indices says which
rows fitted those maps. Native metadata/split/problem remain available and are
NOT deeply immutable. This object does not create a model or training run.
"""
    physical_field: PhysicalField
    coordinates: object
    targets: object
    train_indices: np.ndarray
    preprocessing_state: Mapping[str, object]
    native: object
    split: object | None = None
    problem: object | None = None

    def trace_row(self, index: int) -> dict[str, object]:
        """Explain one retained row from source identity through normalized tensors.

Returns JSON-compatible numeric values for a private/local worked example, with
axis units, physical target and declared fit population. Negative/out-of-range
indices fail rather than quietly tracing a different row. Reading draws no RNG.
"""
        field = self.physical_field
        if index < 0 or index >= len(field.coordinates):
            raise IndexError(index)
        return dict(row=index, source=field.source, coordinate_mode=field.coordinate_mode,
                    source_identity={name: values[index].item() if isinstance(values[index], np.generic)
                                     else values[index] for name, values in field.row_identity.items()},
                    axes=list(field.axes), units=list(field.units),
                    physical_coordinates=field.coordinates[index].tolist(),
                    physical_target=field.targets[index].tolist(), target_transform=field.target_transform,
                    normalized_coordinates=self.coordinates[index].tolist(),
                    normalized_target=self.targets[index].tolist(),
                    fit_policy=self.preprocessing_state["policy"], fit_rows=len(self.train_indices),
                    row_in_fit_population=bool(np.any(self.train_indices == index)))


def _prepare_xyt(inputs: InputSpec, preprocessing: PreprocessingSpec) -> PreparedInput:
    """Use legacy CSV/band-reader policies and their shared window kernel unchanged."""
    from datasets import RadarTimeH5Dataset
    from synthetic_dataset import SyntheticPlasmaTimeDataset

    selection = dict(inputs.selection)
    selection.update(normalize_coords=preprocessing.normalize_coords,
                     normalize_values=preprocessing.normalize_values)
    native = (RadarTimeH5Dataset(inputs.path, **selection) if inputs.source == "radar"
              else SyntheticPlasmaTimeDataset(inputs.path, **selection))
    identities = {"csv_row_position" if inputs.source == "synthetic" else "retained_reader_position":
                  native.source_row_positions}
    for name in (["time_index", "beam_index", "range_index"] if inputs.source == "radar"
                 else ["time_index", "beam_id"]):
        if name in native.df:
            identities[name] = native.df[name].to_numpy()
    field = PhysicalField(inputs.source, "xyt", native.coords_raw, native.values_raw, identities,
                          metadata={"dataframe": native.df, "source_path": str(inputs.path),
                                    "source_selection": dict(inputs.selection)},
                          allow_nonfinite=not bool(selection.get("drop_bad_rows", True)))
    return PreparedInput(field, native.coords, native.values, np.arange(native.n_samples),
                         dict(policy="selected_window", coordinate_scalers=native.coord_scalers,
                              target_scaler=native.target_scaler,
                              normalize_coords=native.normalize_coords, normalize_values=native.normalize_values), native)


def _prepare_xyzt(inputs: InputSpec, preprocessing: PreprocessingSpec) -> PreparedInput:
    """Retain multi-altitude/analytic policies and original split-before-fit order."""
    import torch
    from inr_isr_4d.pfisr import PFISRReadConfig, read_pfisr_4d
    from inr_isr_4d.synthetic import (MovingFeature4D, SyntheticFieldConfig,
                                    SyntheticObservationConfig, generate_synthetic_case)
    from inr_isr_4d.splits import make_group_split, prepare_training_problem

    if inputs.source == "radar":
        native = read_pfisr_4d(Path(inputs.path), PFISRReadConfig(**inputs.selection))
    else:
        field = dict(inputs.field)
        if isinstance(field.get("feature"), Mapping):
            field["feature"] = MovingFeature4D(**field["feature"])
        native = generate_synthetic_case(SyntheticFieldConfig(**field),
                                         SyntheticObservationConfig(**inputs.observation))
    bundle = native.bundle
    split = make_group_split(bundle, **preprocessing.split)
    problem = prepare_training_problem(bundle, split)
    identities = dict(beam_id=bundle.beam_ids, time_id=bundle.time_ids,
                      group_id=bundle.group_ids)
    if inputs.source == "radar":
        identities.update(record_index=native.record_indices, range_index=native.range_indices)
    else:
        # This is the generator's output position, not a fabricated raw radar gate.
        identities["generator_row_position"] = np.arange(bundle.size)
    physical = PhysicalField(inputs.source, "xyzt", bundle.coordinates, bundle.targets,
                             identities, metadata={"bundle": dict(bundle.metadata), "native_case": native})
    return PreparedInput(physical,
                         torch.as_tensor(problem.coordinate_scaler.transform(bundle.coordinates), dtype=torch.float32),
                         torch.as_tensor(problem.target_scaler.transform(bundle.targets), dtype=torch.float32),
                         problem.train_indices,
                         dict(policy="training_split", coordinate_scaler=problem.coordinate_scaler.state_dict(),
                              target_scaler=problem.target_scaler.state_dict(), split=split.state_dict()),
                         native, split, problem)


def prepare_input(inputs: InputSpec, preprocessing: PreprocessingSpec) -> PreparedInput:
    """Prepare one named workflow; source, mode and fit policy are separate choices.

Native errors propagate. Readers never write source files. Only original synthetic
generation/group-split randomness is used. No truth replacement, weighting,
clipping or model/loss/precision selection is added. Old class/module/scaler and
training interfaces remain in place for later results-preserving stages.
"""
    expected = "selected_window" if inputs.coordinates == "xyt" else "training_split"
    if preprocessing.policy != expected:
        raise ValueError(f"{inputs.coordinates} requires explicit {expected} preprocessing.")
    return (_prepare_xyt(inputs, preprocessing) if inputs.coordinates == "xyt"
            else _prepare_xyzt(inputs, preprocessing))
