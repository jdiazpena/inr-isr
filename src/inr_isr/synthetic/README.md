Created: 2026-10-09
Last edited: 2026-10-11
Status: Additive analytic scene generation and truth rendering; no neural workloads

# Structured synthetic scenes

This package adds a smooth sinusoidal interface between two density regions and
a frozen list of positive Gaussian enhancements. All components translate with
one selected x velocity. It reuses `synthetic_plasma.MovingGaussianPatch` and
the existing time-array helper. Existing generator cases/defaults, dataset and
trainers remain available. This is a designed reconstruction control, not fluid
evolution or a model of plasma production.

`field.py` owns physical-coordinate values and first/second log-density
derivatives. `design.py` draws a patch list once using PCG64; save its returned
parameters. `tables.py` produces the existing synthetic CSV interface.
`render.py` streams matched views through system FFmpeg and verifies them with
ffprobe/full decoding. `records.py` saves small command/source/runtime receipts.

From a project source environment, use a fresh output path:

```bash
python -m inr_isr.synthetic generate --recipe scene.json --geometry geometry.csv --output outputs/new_scene
python -m inr_isr.synthetic generate --recipe scene.json --geometry geometry.csv --output outputs/new_window --start-sec 3600 --end-sec 14400
python -m inr_isr.synthetic stills --recipe scene.json --geometry geometry.csv --output outputs/new_stills --times-sec 0 21600 43200
python -m inr_isr.synthetic render --recipe scene.json --geometry geometry.csv --output outputs/new_movie --frame-step-sec 10 --fps 30
```

The saved recipe defines duration/cadence, truth margin, domain, front, realised
patch list, case IDs/velocities and fixed render colour limits. Geometry needs
`beam_index`, `x_km`, `y_km`; other metadata survives in the observation table.
Generation selects original time records inside a requested window and keeps
their IDs; it never redraws the scene. Outputs fail on occupied destinations.
`SyntheticPlasmaTimeDataset` loads the resulting tables. Its existing window
normalization policy is unchanged; loading is not held-out scientific evaluation.

Main generation uses instantaneous samples. `--integration-sec` optionally
averages the underlying linear density once over centred endpoints, then takes
log10. Internal maximum spacing defaults to 0.1 s and is independent of cadence
and movie spacing. Exposures query the saved analytic margins; out-of-range
queries fail without padding. Accuracy must be checked for each new recipe.
Only values and first derivatives are returned for positive exposures; second
derivative truth is currently instantaneous.

Default truth sums every Gaussian. Rendering may omit whole patches farther
than ten Gaussian widths from every grid point; the evaluator returns the
absolute density omission bound. Static views save numerical plot inputs.
MP4s and stills display log10(Ne/m^-3), with one shared logarithmic density scale
across cases and times. The recipe's positive density bounds are converted to
log10 colour limits; receipts record the display transform and limits. Saved
numerical plot inputs retain their original values.
MP4s show simultaneous cases with fixed colour limits, coordinates, locations,
elapsed time and playback mapping. Software encoding uses CPU; no neural model
is constructed. Movie exports require `ffmpeg` (including `ffprobe`) installed
using the container's system package manager; no executable bundles belong here.

Run focused non-neural tests directly, without a broad training-fixture suite:

```bash
python tests/test_structured_scene_generation.py
```

These check finite-difference derivatives, rigid translation and density
averaging against independent adaptive quadrature. They establish generator
correctness for the fixture, not physical realism or reconstruction performance.
