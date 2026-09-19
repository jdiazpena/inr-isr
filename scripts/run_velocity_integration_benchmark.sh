#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/project_environment.sh" "$@"
set -u
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SCOPE="${SCOPE:-pilot}"
STAGE="${STAGE:-generate}"
NUM_STEPS="${NUM_STEPS:-15000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/velocity_integration_benchmark}"

"$INR_ISR_PYTHON" scripts/synthetic_velocity_integration_benchmark.py \
  --scope "$SCOPE" \
  --stage "$STAGE" \
  --num_steps "$NUM_STEPS" \
  --output_root "$OUTPUT_ROOT" \
  "$@"
