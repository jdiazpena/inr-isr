# Project Handoff

This document is the durable starting point for a new Codex CLI session. It records
the current scientific contract, repository state, important decisions, verification,
and the next work to undertake. Read this file first, then follow the reading order in
the root `README.md`.

## Project Location

The active working copy is:

```text
/home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d
```

This directory is intentionally a local working copy and is not a Git repository.
The apparent alternate paths seen previously are symlinks to different drives, not
different projects.

The WSL Conda installation is `/home/jdiaz/miniconda3`. Interactive terminals already
show `(base)` and therefore do not need manual activation. Repository shell launchers
activate that installation themselves so they also work from a fresh shell.

## Scientific Objective

Reconstruct a continuous three-dimensional, time-dependent plasma electron-density
field from sparse ISR measurements using an implicit neural representation. The main
model remains a SIREN-style INR. The interpolation should fit measured data while
using soft spatial and temporal priors consistently and should eventually expose both
the interpolation and a reliability map to users.

This project must not become a physics-informed neural network (PINN); that is reserved
for a separate proposal. Physics and geometry may inform soft priors, sampling,
validation, and uncertainty analysis without imposing a new governing-equation loss.

## Current Scientific Contract

- The primary INR is the sine-activated `MLPINR` in `src/models.py`.
- Current defaults are 256 hidden features, 3 hidden layers, and SIREN frequencies
  `first_omega_0=5` and `hidden_omega_0=5`.
- The accepted synthetic reconstruction analyzer is
  `src/synthetic_analyze_reconstruction.py`, which evaluates linear electron-density
  errors. Historical misspelled module names remain compatibility aliases only.
- Synthetic targets for an integrated radar product are integration-averaged truth,
  not instantaneous midpoint truth.
- Radar integration time describes the available measurement product. Smearing caused
  by upstream radar processing is not an INR performance penalty and should not drive
  model selection. Smaller integration products may be requested or reprocessed when
  scientifically needed, but that is outside the reconstruction model.
- Spatial regularization must respect ISR geometry and anisotropy. Vertical (`z`)
  behavior must be assessed separately from horizontal (`x`, `y`) behavior rather
  than assuming isotropy.
- Curvature and gradient diagnostics are retained because a regularization term whose
  curvature collapses toward zero may no longer be learning useful structure.
- Reliability maps will use either MC dropout with geometry/support features or
  conformal prediction. Ensembles were discussed in the proposal but are explicitly
  out of scope for the implementation.

## Validation Commitments

Evaluation must go beyond random point holdout. The intended protocol includes:

- withheld beams to test spatial interpolation;
- withheld time intervals to test temporal interpolation;
- joint beam/time withholding for the hardest unsupported regions;
- separate reporting by beam support and distance to measurement support;
- reliability coverage and calibration, not only point-estimate error;
- visual products that show the reconstruction and reliability map together, including
  explicit low-trust regions rather than hiding them.

Synthetic cases remain useful for controlled identifiability and failure analysis,
but they are not sufficient evidence by themselves. The next major scientific stage
should pair controlled synthetic or semi-synthetic cases with a well-supported real
42-beam ISR event.

## Synthetic Experiment Decisions

The expanded sequential benchmark contains:

- beam supports: 42, 23, and 11 beams;
- horizontal speeds: 0.36 and 2.00 km/s;
- integration products: 1, 2, and 5 minutes;
- training modes: data-only and `xy030_t030` regularization;
- seed: 0;
- total: 18 generated data cases and 36 model runs.

The launcher is deliberately simple and foreground-only: one model runs after another
and `tqdm` progress remains visible in the terminal. Do not reintroduce `nohup`, `tee`,
parallel execution, or background process management unless explicitly requested.

Flow reversal or strong shear is an important rare stress case: an auroral arc may
divide the field of view so that opposite regions move in opposite directions. Keep
this as a separate targeted experiment rather than adding it as another factor to the
main 36-run benchmark. Future controlled cases should also cover direction variation,
vertical structure, multiple structures, and support-boundary behavior.

## Repository State

The repository is organized into `config/`, `docs/`, `outputs/`, `scripts/`, `src/`,
and `tests/`. Training behavior was refactored into shared modules while preserving
the SIREN optimization contract for both synthetic and real radar data:

- `src/training_engine.py`: shared optimization loop;
- `src/training_common.py`: batching, metrics, collocation, curvature, adaptive
  weighting, diagnostics, and evaluation;
- `src/training_config.py`: typed configuration and optional JSON loading;
- `src/synthetic_train_3d.py`: synthetic adapter and plots;
- `src/train_radar_3d_window_reference_reg_diagnostic.py`: real-radar adapter and plots;
- `src/amisr_h5_reader_3d.py`: local real-data HDF5 reader, making this copy
  self-contained.

Existing generated results were preserved under `outputs/`. No long benchmark was
started by Codex during the refactor.

## Verified Behavior

Run the full verification suite with:

```bash
cd /home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d
bash scripts/run_tests.sh
```

At the last verification, all 13 tests passed, including SIREN initialization,
analytic curvature, configuration precedence, synthetic physics and integration,
42/23/11-beam geometry, a minimal AMISR HDF5 fixture, the real radar dataset, wrapper
command parsing, historical analysis aliases, and benchmark scope counts.

Four-step synthetic and tiny-HDF5 real-radar comparisons were byte-identical in their
training histories before and after refactoring. Final model tensors were also exactly
equal. The accepted dense reconstruction analyzer ran successfully with gradients.

## Commands

Run the 1/2/5-minute benchmark from the project root:

```bash
bash scripts/run_overnight_125_benchmark.sh
```

It resumes completed work by skipping complete models and restarts an incomplete model
directory cleanly. It runs sequentially and displays live progress.

Use an explicit configuration file with a trainer as follows; explicit command-line
arguments override JSON values:

```bash
python src/synthetic_train_3d.py \
  --config config/training_defaults.json \
  [other arguments]
```

## Recommended Next Work

1. Analyze the completed 1/2/5-minute, 11/23/42-beam benchmark using reconstruction
   error against each product's correct integration-averaged truth. Do not rank SIREN
   by unavoidable upstream smearing.
2. Use the benchmark to identify where support geometry and the current soft priors
   help or fail, including curvature-collapse diagnostics.
3. Add one isolated flow-reversal/shear stress case without expanding the main
   factorial benchmark.
4. Select one well-supported real 42-beam ISR event and define beam/time withholding
   before tuning further regularization.
5. Replace sparse manual lambda trials with a defensible protocol: nested validation
   or adaptive weighting, evaluated on withheld beam/time tasks rather than training
   loss alone.
6. Implement and compare reliability maps using MC dropout plus support/geometry
   features and conformal calibration. Select based on empirical coverage, sharpness,
   spatial failure localization, and operational usefulness.
7. Only after establishing the SIREN baseline and validation protocol, compare focused
   INR alternatives such as FINER or K-Planes under identical splits and budgets.

## Important Boundaries

- Preserve the SIREN baseline and numerical behavior unless a change is explicitly an
  experiment with its own configuration and comparison.
- Preserve compatibility entry points used by previous runs.
- Do not delete or overwrite historical outputs merely to tidy the repository.
- Do not treat synthetic data as the final scientific validation.
- Do not use ensembles for uncertainty and do not turn the model into a PINN.
- Keep user-facing execution simple, sequential, foregrounded, and observable.

## Supporting Documents

- `SCIENTIFIC_PIPELINE.md`: mathematical and computational pipeline.
- `REFACTORING_NOTES.md`: code changes and behavior-equivalence evidence.
- `OVERNIGHT_125_BENCHMARK.md`: current overnight benchmark runbook.
- `SYNTHETIC_VELOCITY_INTEGRATION_BENCHMARK.md`: full benchmark design.
- `CODEX_INR_PLAN.md`: implementation inventory and original regularization plan.
- `CODEX_SYNTHETIC_SWEEP_REPORT.md`: initial synthetic sweep findings.

The original attached request is archived at:

```text
docs/archive/original_request.txt
```

The research plan and long-form synthetic pilot report are retained entirely inside
the Linux repository:

```text
docs/reports/inr_isr_research_plan.pdf
docs/reports/inr_isr_research_plan.tex
docs/reports/inr_synthetic_pilot_results/
```

The pilot report directory includes its PDF, LaTeX source, figure-generation script,
and all referenced figures. No Windows-mounted path is required to read or rebuild
these documents.

## Codex Session Recovery

The durable Codex CLI session associated with this project and its research history is:

```text
019f62f8-83a6-7ef0-b22c-056e019a5c96
```

Resume it from any directory with:

```bash
codex resume 019f62f8-83a6-7ef0-b22c-056e019a5c96
```

Running plain `codex` creates a separate session; it does not delete this one. If it
is no longer the most recent session, use the explicit identifier above or locate it
with `codex resume --all`. The underlying local session must still exist under the
same Linux user's `~/.codex/sessions/` directory.
