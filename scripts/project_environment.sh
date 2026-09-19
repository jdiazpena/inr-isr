#!/usr/bin/env bash
# Source this helper before starting a workload. Never fall back to base.
# PYTHON_BIN remains an optional override, but must belong to inr-isr.

if ! source /home/jdiaz/miniconda3/etc/profile.d/conda.sh; then
  echo "Cannot load Conda; no INR--ISR workload was started." >&2
  return 1
fi
if ! conda activate inr-isr; then
  echo "Required environment inr-isr is unavailable; no fallback is allowed." >&2
  return 1
fi

INR_ISR_ENV_PREFIX="/home/jdiaz/miniconda3/envs/inr-isr"
INR_ISR_PYTHON="${PYTHON_BIN:-$INR_ISR_ENV_PREFIX/bin/python}"
if ! "$INR_ISR_PYTHON" -I -c 'import sys; from pathlib import Path; expected=Path(sys.argv[1]).resolve(); assert Path(sys.prefix).resolve()==expected, "Interpreter is outside inr-isr: " + sys.executable' "$INR_ISR_ENV_PREFIX"; then
  echo "Invalid project interpreter: $INR_ISR_PYTHON; no workload was started." >&2
  return 1
fi
export INR_ISR_ENV_PREFIX INR_ISR_PYTHON

# Verification mode exits the calling launcher before files/training are created.
if [[ "${1:-}" == "--check-environment" ]]; then
  "$INR_ISR_PYTHON" -I -c 'import sys; print("Interpreter:", sys.executable); print("Python:", sys.version.split()[0])'
  exit 0
fi
