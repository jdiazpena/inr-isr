#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/project_environment.sh" "$@"
set -u
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_ROOT="$(mktemp -d /tmp/inr-radar-tests.XXXXXX)"
trap 'rm -rf "$TEMP_ROOT"' EXIT

cd "$PROJECT_ROOT"
export MPLCONFIGDIR="$TEMP_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TEMP_ROOT/pycache"
export XDG_CACHE_HOME="$TEMP_ROOT/cache"

"$INR_ISR_PYTHON" -m compileall -q src scripts tests
"$INR_ISR_PYTHON" -m unittest -v \
  tests/test_characterization.py \
  tests/test_velocity_integration_benchmark.py

cd "$TEMP_ROOT"
"$INR_ISR_PYTHON" "$PROJECT_ROOT/tests/test_synthetic_plasma.py"
"$INR_ISR_PYTHON" "$PROJECT_ROOT/tests/test_synthetic_dataset.py"

echo "All INR radar tests passed."
