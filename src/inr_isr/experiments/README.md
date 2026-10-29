Created: 2026-10-26
Last edited: 2026-10-28
Status: Explicit development, reconstruction and separate assessment routes; neural jobs are operator-run

# Explicit development experiments through the shared trainer

`python -m inr_isr.experiments train --contract CASE.json --output NEW_FOLDER`
loads a checksummed case and calls the existing `training_engine.train_window`
with the synthetic dataset/plot adapters. It adds no optimizer loop. `--smoke`
selects three updates; `--cpu` is an explicit alternative to required CUDA.
`analyze --contract CASE.json --run TRAIN_FOLDER --output NEW_FOLDER` evaluates
the best post-update development checkpoint against its frozen analytic scene.
Both destinations must be fresh. These commands execute neural work only when
the operator invokes them; importing the package does not launch an experiment.

The contract declares disjoint train/development/reserved beam IDs, checksummed
training/development CSVs, fixed probe and evaluation-coordinate CSVs, scene
definition, numerical settings and the checkpoint schedule. Paths resolve
relative to the contract. Supply a training-only CSV: its normal existing dataset
fits normalization solely on those training rows. Reserved targets are not inputs
to this module. Full-domain analytic analysis is development information, not
independent final evidence.

`DevelopmentMonitor` is an optional argument to the shared engine. Default callers
retain their original dataset, random diagnostic sampling, controller/update
sequence, legacy best-loss checkpoints and terminal display. Experiment callers
use saved fixed probes without drawing from the initialization RNG, and evaluate
development observations after optimizer updates. Selection is
`sqrt(mean(per-location MSE in dex))`; exact ties retain the earlier checkpoint.
Native spatial/temporal curvature and fixed/adaptive weight policies remain in
the shared engine. A data-only contract disables their contribution but retains
fixed derivative diagnostics. The concise experiment progress line leaves the
full diagnostic histories in CSVs.

Analysis uses existing checkpoint construction, prediction, affine transforms
and autograd kernels, extending the latter to physical Hessian components.
It saves logarithmic maps, field/observation/derivative errors, support/distance
labels, a space-time cut and truth-assisted front/patch profile measurements.
Profile matching uses the known synthetic geometry/background and reports
missing/multiple crossings. Flat patch cores use the half-height interval centre,
not an arbitrary argmax. These are diagnostic associations, not a general feature
detector for real radar. A failed/nonfinite prediction stops analysis rather than
silently removing queries.

`tests/test_development_experiment.py` exercises scoring, contracts, crossing
failures, physical chain factors and selection/analysis plumbing using mathematical
functions without neural layers, parameters or optimization. Those checks cannot
establish neural training parity or reconstruction accuracy. The operator must
run the CUDA smoke and inspect real outputs before scientific interpretation.


## Final-update reconstruction and separate withholding assessment

`shared-xyt-reconstruction-1` requires every declared beam and cadence timestamp
and saves the final completed update. `shared-xyt-withholding-1` explicitly
partitions a checksummed complete source table into training and withheld rows.
Declare either removed beam indices or interior timestamps. It also saves the
final update: withheld targets never select weights, stopping or checkpoints.
The existing development schema retains its original selection behaviour.

The withholding schema checks exact source coverage, complete disjoint row
partitions, geometry roles and time bounds. Normalization uses training rows
only. Analysis reports training and withheld errors, refreshes spatial support
and time gaps from retained observations, and marks exclusions in plots.
Cached all-observation support labels are not reused for reduced inputs.
Both modes use the same native model, objective and optimizer stages; the loss
terms are configured independently of the observation assessment design.

Partition, final-save and analytic-function analysis checks are in
`tests/test_withholding_experiment.py`; no neural model or optimizer is created
by those tests. Actual training and prediction still require operator launch.

## Native radar reconstruction and withheld-measurement assessment

`shared-xyt-radar-reconstruction-1` uses the native HDF reader and every retained
observation in its declared UTC window. Its completeness guard still rejects
undeclared beam or time removal. Native irregular midpoint times, integration
intervals and original invalid-density exclusions remain explicit.

`shared-xyt-radar-withholding-1` first checks that complete native source, then
validates an exact beam-only or time-record-only partition. Declare native beam
indices or interior source record indices and their local seconds. Training
uses the same radar dataset class and normalization kernel, refitted on retained
rows before constructing the model. The source table and withheld values are
integrity/assessment inputs; neither fits the final training scalers or selects
weights, stopping or a checkpoint. Both routes use the existing shared trainer,
reported-error loss options, native initializer and fixed final-update policy.

Assessment recomputes distances, hulls and gaps from retained native observations;
it does not assume a uniform cadence. It adds withheld residual tables, both
standardized residual scales when configured, an assessment plot and exclusion
markers. Withheld values are noisy exposure-averaged estimates, not plasma
truth or an error estimate for the all-observation model. Analytic tests in
`tests/test_radar_withholding.py` construct no neural model or optimizer.
