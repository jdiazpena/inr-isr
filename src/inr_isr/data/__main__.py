"""Inspect one JSON-configured input locally; no training or loss is selected.

Run ``python -m inr_isr.data --config case.json --output fresh-directory`` with
canonical src on PYTHONPATH. Separate training_recipe provenance may accompany
input/preprocessing in a teaching config, but is never executed by this inspector.
Numerical inspection files can contain private radar values: keep outputs local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .inputs import InputSpec, PreprocessingSpec, prepare_input


def main(argv: list[str] | None = None) -> int:
    """Validate a bounded config, prepare observations and save one answered row trace.

Reject occupied destinations or any destination inside a frozen reference packet
before reading data. A failed preparation does not create an output directory.
Generated JSON records runtime and config; scientific input/source identities are
recorded by the project verification harness, not asserted by this general CLI.
No dependency upgrades or scientific product claims are performed.
"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or any((p / "REFERENCE_MANIFEST.json").exists()
                              or ((p / "BASELINE_MANIFEST.json").exists()
                                  and output.is_relative_to(p / "baseline"))
                              for p in [output, *output.parents]):
        raise FileExistsError("Use a fresh output outside frozen reference/baseline packets.")
    config = json.loads(args.config.read_text())
    if not {"input", "preprocessing"} <= set(config) or set(config) - {"input", "preprocessing", "training_recipe"}:
        raise ValueError("Config requires input/preprocessing; only separate training_recipe provenance is optional.")
    prepared = prepare_input(InputSpec(**config["input"]), PreprocessingSpec(**config["preprocessing"]))
    trace = prepared.trace_row(args.row)
    # Validate JSON before creating a directory (legacy nonfinite data can be retained).
    payload = json.dumps(dict(executable=sys.executable, config=config, observations=len(prepared.coordinates),
                              preprocessing_state=prepared.preprocessing_state, row_trace=trace),
                         indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "input_inspection.json").open("x") as stream:
        stream.write(payload)
    print(json.dumps(dict(observations=len(prepared.coordinates), output=str(output)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
