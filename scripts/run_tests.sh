#!/usr/bin/env bash
set -eo pipefail

source /home/jdiaz/miniconda3/etc/profile.d/conda.sh
conda activate base
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_ROOT="$(mktemp -d /tmp/inr-radar-tests.XXXXXX)"
trap 'rm -rf "$TEMP_ROOT"' EXIT

cd "$PROJECT_ROOT"
export MPLCONFIGDIR="$TEMP_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TEMP_ROOT/pycache"
export XDG_CACHE_HOME="$TEMP_ROOT/cache"

python -m compileall -q src scripts tests
python -m unittest -v \
  tests/test_characterization.py \
  tests/test_velocity_integration_benchmark.py

cd "$TEMP_ROOT"
python "$PROJECT_ROOT/tests/test_synthetic_plasma.py"
python "$PROJECT_ROOT/tests/test_synthetic_dataset.py"

echo "All INR radar tests passed."
