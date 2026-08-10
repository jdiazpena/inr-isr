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

The verified Python dependencies are recorded in `requirements.txt`. The existing
WSL installation uses `/home/jdiaz/miniconda3`; shell launchers activate its `base`
environment automatically.

## Main Workflows

Run commands from the new project root:

```bash
cd /home/jdiaz/postdoc/inr-isr-4d
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
