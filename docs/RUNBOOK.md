Created: 2026-08-10
Last edited: 2026-08-10
Status: Active operational runbook

# 4D experiment runbook

## Environment and tests

```bash
cd /home/jdiaz/postdoc/inr-isr-4d
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests tests_4d
```

Optional local installation, using the already installed dependencies:

```bash
python -m pip install . --no-deps --no-build-isolation
```

Without installation, prefix runner commands with `PYTHONPATH=src`.

## Stages

The runner supports `manifest`, `generate`, `train`, `calibrate`, `evaluate`,
`summarize`, `check`, and `all`. Calibration consumes a fixed trained checkpoint and
does not examine test targets. Evaluation consumes the saved calibration artifact.
Completed stages are validated and skipped; failures produce a structured failure
artifact.

```bash
PYTHONPATH=src python -m inr_isr_4d.runner MANIFEST all --case-id CASE_ID
PYTHONPATH=src python -m inr_isr_4d.runner MANIFEST train --case-id CASE_ID --resume
```

Run directories refuse collisions. To restart rather than resume, choose a new
attempt name so the earlier run remains intact:

```bash
PYTHONPATH=src python -m inr_isr_4d.runner MANIFEST all --case-id CASE_ID --restart --attempt-id restart-01
```

Explicit command-line overrides are applied only when supplied. Zero is a valid
seed, and every `--set` value is parsed as JSON before strict configuration
validation:

```bash
PYTHONPATH=src python -m inr_isr_4d.runner MANIFEST all --case-id CASE_ID --seed 0 --set runtime.device='"cuda"' --set collocation.pool_size=250000
```

Collocation bounds are normalized four-vectors in `(x, y, z, t)` order and may be
restricted within the training-defined `[-1, 1]` coordinate domain:

```bash
PYTHONPATH=src python -m inr_isr_4d.runner MANIFEST all --dry-run --set 'collocation.domain_lower=[-0.8,-1.0,-1.0,-1.0]' --set 'collocation.domain_upper=[0.8,1.0,1.0,1.0]'
```

## Dry-run contract

`--dry-run` reads and validates configurations, inspects actual HDF5 metadata,
regenerates deterministic in-memory case structure, calculates split/collocation
counts, and prints device, tensor, command, and output plans. It creates no output
directory or run artifact.

```bash
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/smoke.json all --dry-run
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/pilot.json all --dry-run
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/long.json all --dry-run
```

## Study sequence

1. Run and check the CPU smoke manifest.
2. On the 16 GB GPU, dry-run smoke and execute a selected derivative case to verify
   CUDA device identity, memory telemetry, and checkpoint resume.
3. Execute the pilot manifest only after inspecting its expansion. Use pilot timing
   and peak-memory results to adjust batch/microbatch sizes, not scientific factors.
4. Freeze the long manifest and its Git commit before launch.
5. Launch or resume the long study with the exact command below only after Joaquín
   approves the frozen pilot review.

Prepared later long-study command (not executed during implementation):

```bash
cd /home/jdiaz/postdoc/inr-isr-4d
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/long.json all
```

## 16 GB GPU verification command

The current sandbox does not expose CUDA. On the target GPU, first run:

```bash
cd /home/jdiaz/postdoc/inr-isr-4d
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/pilot.json all --dry-run --case-id "pilot support aware high collocation"
PYTHONPATH=src python -m inr_isr_4d.runner config/4d/manifests/pilot.json all --case-id "pilot support aware high collocation"
```

Confirm the resolved device is CUDA and inspect `history.jsonl` peak memory before
authorizing a broader pilot. If an `auto` run previously resolved to CPU, use a new
attempt and set the manifest device explicitly to `cuda`; requested unavailable CUDA
is an error and is never silently changed.
