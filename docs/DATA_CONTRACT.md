Created: 2026-08-10
Last edited: 2026-08-10
Status: Active data and split contract

# Data contract

## Coordinate and target semantics

`FieldBundle4D.coordinates` has physical columns `(x_km, y_km, z_km, t_sec)` in
that exact order. `targets` is a two-dimensional matrix; current electron-density
experiments use one log10-density column. Every observation has explicit beam,
time-block, and joint-group identifiers. A full-field bundle is immutable and is
not exposed through a misleading sample index.

`SampleDataset4D` follows ordinary PyTorch semantics: its length is the number of
observations and one item is one observation. Full-array access remains explicit on
the bundle.

## Split-before-preprocessing rule

Group assignments are constructed before fitting target preprocessing or using
held-out targets for any selection. Supported group units are beam, time block, and
joint beam/time. Supported withholding strategies are deterministic random and
clustered selection. Requested validation, calibration, and test group counts are
exact; geometric radius expansion cannot silently increase them.

The four disjoint roles are:

- training: model fitting and fitted preprocessing;
- validation: model-development decisions only;
- calibration: conformal score calibration only;
- test: one untouched final empirical evaluation.

The saved split contains exact indices, group identifiers, observation/group
counts, strategy, unit, and seed. `TrainingProblem4D` contains only training indices
for optimization even though the immutable source bundle remains available to later
separate stages.

## Normalization

Coordinates and targets use a per-column affine mapping to `[-1, 1]` fitted on
training observations only. Values outside the training minimum/maximum are not
clipped; calibration and test extrema may legitimately transform outside `[-1, 1]`.
The fitted state is saved in both run metadata and every complete checkpoint.

## External PFISR data

The external HDF5 files are read-only inputs and are never copied into or committed
to this repository. Their exact reader/filter/integration-duration contract will be
added and verified as a separate adapter milestone.
