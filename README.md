Created: Unknown (historical repository document)
Last edited: 2026-09-14
Status: Active overview; latest-stable inr-isr environment supersedes historical base setup

# INR ISR 4D Reconstruction and Split Conformal Evaluation

## Evidence and scope boundary

This repository began as a literal copy of the complete working 3D implementation
at `/home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d`. The copied 3D modules,
tests, configurations, and scripts remain the working compatibility baseline; they
were not recreated or generalized before 4D work began.

The additive package under `src/inr_isr_4d` implements the next UAI-stage research
capability: 4D `(x, y, z, t)` neural-field reconstruction, independent derivative
collocation, leakage-safe grouped split conformal prediction, and reproducible
benchmark operations. The inspected repository
`/home/jdiaz/codex/postdocanid/inr-isr` was used only as a component-level historical
reference; its standalone trainer architecture was not copied.

The implemented curvature terms are derivative/smoothness priors. This repository
does **not** implement plasma-transport equations, precipitation models,
auroral-camera assimilation, or the complete physics-informed FONDECYT proposal.
Those are future proposed developments. Passing tests and smoke runs establish
executable capability, not scientific superiority or validated plasma physics.

This project reconstructs continuous ionospheric plasma fields from sparse radar
measurements using implicit neural representations (INRs). It contains workflows for
both synthetic experiments and real radar windows.

The repository is organized by responsibility so that source code, executable
workflows, documentation, tests, and generated results are not mixed together.

## Project Map

| Path | Purpose |
|---|---|
| `src/` | Scientific implementation: models, datasets, synthetic generation, training, and reconstruction analysis |
| `scripts/` | Commands that coordinate experiments, parameter sweeps, benchmarks, and repeated radar windows |
| `config/` | Home for configuration files as hard-coded experiment settings are externalized |
| `tests/` | Numerical characterization, real-reader, benchmark, and synthetic smoke tests |
| `docs/` | Research plans, benchmark protocols, findings, and runbooks |
| `outputs/` | Generated datasets, checkpoints, histories, figures, tables, and archived results |

Raw radar data are not copied into this repository. Real-data workflows receive the
HDF5 input path as a command-line argument or through their current defaults.

Historical verified dependencies are recorded in `requirements.txt`; those older
pins are not the current latest-stable environment specification. The existing WSL
installation uses `/home/jdiaz/miniconda3`; all four shell launchers now activate
`inr-isr` through `scripts/project_environment.sh`, validate the interpreter and
never fall back to base. The environment uses Python 3.14.7 and latest stable
dependencies installed on 2026-09-14. Recorded versions and compatibility limits
are in `/home/jdiaz/projects/postdocuai/docs/topics/project_environment/README.md`.
The old `np.trapz` test calls were changed to `np.trapezoid` without changing
their inputs, assertions or tolerances. Under the recorded `inr-isr` environment,
the current GPU-repair source-bound suite passes 181 tests; bounded
four-workflow CUDA training/reload/query preservation is separately recorded
in the documentation workspace. Neither installation nor tests imply
scientific validation.

Check any shell launcher's environment without running its workload:

```bash
bash scripts/run_tests.sh --check-environment
```

Direct Python workflows require `conda activate inr-isr` or its explicit
interpreter. Neural-network training and inference require CUDA by default and
fail clearly when it is unavailable. CPU remains supported only when selected
explicitly; CPU-side data loading, plotting and serialization are normal and do
not imply that the model executed on the CPU.

## Main Workflows

### Explicit xyt/xyzt reconstruction front door

`python -m inr_isr.workflow` now has `train` and `query` commands. A JSON file
separately declares `input` (synthetic or radar and named xyt/xyzt axes),
`preprocessing` (the existing selected-window or training-split fit),
`training_recipe` (one existing scientific policy), and an optional physical
`query`. The named recipes are `active_xyt_synthetic`, `active_xyt_radar`, and
`native_xyzt`; the latter accepts either existing synthetic or radar xyzt
source. An input source does not silently select a loss. The older fixed and
reference-window 3D trainers keep their original commands and are not renamed
as new recipes.

Use the guarded project interpreter and a fresh output directory. These commands
use CUDA by default; do not add a CPU selector unless CPU execution is the
deliberate experiment:

```bash
/usr/bin/python3 -B /home/jdiaz/projects/postdocuai/docs/topics/project_environment/run_project.py -m inr_isr.workflow train --config /home/jdiaz/postdoc/inr-isr/outputs/refactor_candidate/r6-20260918-01/interface1/S3-L/attempt2/workflow.json --output /home/jdiaz/postdoc/inr-isr/outputs/learning/readme-r6-xyt-synthetic-01
/usr/bin/python3 -B /home/jdiaz/projects/postdocuai/docs/topics/project_environment/run_project.py -m inr_isr.workflow query --config /home/jdiaz/postdoc/inr-isr/outputs/refactor_candidate/r6-20260918-01/interface1/S3-L/attempt2/workflow.json --checkpoint /home/jdiaz/postdoc/inr-isr/outputs/learning/readme-r6-xyt-synthetic-01/training/model_final.pt --output /home/jdiaz/postdoc/inr-isr/outputs/learning/readme-r6-xyt-synthetic-grid-01
```

Those displayed learning-output paths are examples, not previously executed
folders. The four **executed GPU-default** synthetic/actual xyt/xyzt train,
reload, measured and 4,355-grid routes, with worked explanations and exact
results, are in
`/home/jdiaz/projects/postdocuai/docs/topics/gpu_repair_handoff/examples/`.
The actual-radar guides include tested two- and five-minute-product variations.
Change the JSON and use a new output folder for a new experiment. Direct query
returns log10 electron density at physical coordinates; it does not perform
conformal calibration or assert interpolation accuracy.

For the explicit CPU alternative, set
`training_recipe.parameters.cpu=true` in an xyt configuration or
`training_recipe.config.runtime.device="cpu"` in an xyzt configuration. For a
query, set `query.device="cpu"` or pass `--device cpu`, but do not provide
conflicting choices. Training/query outputs include requested and resolved
device, model and batch device, GPU identity when applicable, and precision
metadata; inspect those records instead of inferring device use from speed.

Run commands from the new project root:

```bash
cd /home/jdiaz/postdoc/inr-isr
```

### Additive 4D manifests

Inspect all planned smoke cases without creating artifacts:

```bash
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/smoke.json all --dry-run
```

Execute the deliberately bounded CPU smoke study:

```bash
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/smoke.json all
```

Pilot and long manifests are committed for later use. They must be dry-run before
training; this implementation goal does not launch them. Use `--case-id` to select a
single deterministic case, and `--resume` to continue an incomplete checkpoint.
An explicit restart requires a new `--attempt-id`, preserving historical results.

### Overnight 1/2/5-minute synthetic benchmark

```bash
bash scripts/run_overnight_125_benchmark.sh
```

This runs 42-, 23-, and 11-beam cases at 0.36 and 2.00 km/s for 1-, 2-, and
5-minute integration products. The data-only and `xy030_t030` models are trained
sequentially, giving 36 total model runs.

### Original velocity/integration benchmark

```bash
SCOPE=pilot STAGE=all bash scripts/run_velocity_integration_benchmark.sh
```

### Synthetic regularization sweep

```bash
bash scripts/run_synthetic_regularization_sweep.sh
```

### Tests

```bash
bash scripts/run_tests.sh
```

The complete copied and additive suite can also be run directly:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests tests_4d
```

The test launcher isolates synthetic smoke outputs in a temporary directory.

## Reading Order

For a first pass through the code:

1. Read `docs/PROJECT_HANDOFF.md` for the current state, decisions, and next work.
2. Read `docs/SCIENTIFIC_PIPELINE.md` for the current SIREN and reconstruction workflow.
3. Read `src/models.py` for the SIREN-style INR definition.
4. Read `src/synthetic_plasma.py` and `src/synthetic_dataset.py` for the synthetic field and observations.
5. Read `src/synthetic_train_3d.py` for the synthetic data adapter and plots.
6. Read `src/training_engine.py` and `src/training_common.py` for optimization, curvature, and diagnostics.
7. Read `src/synthetic_analyze_reconstruction.py` for the accepted physical-unit reconstruction analysis.
8. Read `scripts/synthetic_velocity_integration_benchmark.py` to see how complete experiments are assembled.
9. For real radar, read `src/amisr_h5_reader_3d.py` followed by `src/datasets.py`.
10. For additive 4D work, read `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`,
    `docs/CONFORMAL_PROTOCOL.md`, and `docs/RUNBOOK.md`, then `src/inr_isr_4d`.

Compatibility aliases retain the historical `reconsturction` spelling, but active
workflows use the corrected `reconstruction` module names.

## Output Policy

`outputs/` is generated state, not source code. Do not use files there as hidden
configuration inputs unless a workflow explicitly documents that dependency.
Root-level artifacts from earlier work were preserved in
`outputs/archive_root_analysis/`; regularization comparison products live in
`outputs/comparison/`, and historical sweep logs live in `outputs/logs/`.
