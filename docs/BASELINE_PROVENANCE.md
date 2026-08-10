Created: 2026-08-10
Last edited: 2026-08-10
Status: Verified copied 3D foundation

# Baseline Provenance

## Exact source and target

- Source working tree: `/home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d`
- Source parent Git commit: `0b5e104ef6acac8efeec0ade6fedc82a1bf8ce6a`
- Target repository: `/home/jdiaz/postdoc/inr-isr-4d`
- Existing 4D reference, read-only: `/home/jdiaz/codex/postdocanid/inr-isr`
- Existing 4D reference commit: `62b0093f902b3770310659b233670580e716ea88`

The source directory had tracked modifications and untracked implementation and
documentation files. The target was copied from the complete current working tree,
not reconstructed from the parent commit. No source file was cleaned, reset, or
modified.

## Explicit copy exclusions

Only these categories were excluded:

- source Git history (`.git`);
- generated `outputs/`;
- Python and tool caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`,
  `.ruff_cache`);
- local virtual environments (`.venv`, `venv`).

The source, tests, configurations, scripts, runbooks, reports, and supporting
documents outside those categories were copied. A checksum-aware `rsync` dry run
after the copy reported no differences under this exclusion policy.

## Environment

- Python: `3.11.8`
- Python executable: `/home/jdiaz/miniconda3/bin/python`
- PyTorch: `2.11.0+cu128`
- NumPy: `1.26.4`
- CUDA visible in the current execution environment: `false`
- GPU verification status: pending on the target 16 GB GPU environment

## External data identity

The data remain external and were not copied into Git.

| Product | Bytes | SHA-256 |
|---|---:|---|
| `20120122.001_lp_2min-fitcal.h5` | 748,348,215 | `c4dceafc948edccb03c5e33da09804d797685c3897c80f6f3163c4cc82fdddd7` |
| `20120122.001_lp_5min.h5` | 223,201,872 | `a336604f4b7c6411fe93e97847be479d00818fc4dd7867adc20685f61d32e48d` |

## Baseline verification

Read-only source verification before copying:

- Working 3D source: 13 tests passed.
- Existing 4D/conformal reference: 31 tests passed.

Copied-target verification before any 4D implementation:

- Unchanged copied 3D suite: 13 tests passed.
- Source-documented bounded CPU smoke:
  `SCOPE=smoke STAGE=all NUM_STEPS=20 ... --cpu`.
- Smoke result: two generated cases, four trained models, four dense analyses,
  and curvature-health checks completed with exit code 0.
- Smoke artifacts were isolated under `/tmp/inr-isr-4d-baseline-smoke` and were
  not added to the target repository.

This establishes executable capability of the copied 3D foundation. It does not
claim independent scientific validation or validate future 4D behavior.
