"""Separate xyt results commands; model execution is an explicit user action."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from .common import utc, write_json
from .config import load_case


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "train", "analyze", "final-evaluate", "verify-smoke"):
        item = subcommands.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--output", required=True, type=Path)
        item.add_argument("--context", type=Path)
        if name != "prepare":
            item.add_argument("--data", required=True, type=Path)
        if name in {"analyze", "final-evaluate", "verify-smoke"}:
            item.add_argument("--checkpoint", required=True, type=Path)
        if name == "train":
            item.add_argument("--smoke", action="store_true")
        if name == "final-evaluate":
            item.add_argument("--freeze", required=True, type=Path)
    return command


def main():
    args = parser().parse_args()
    if args.output.exists() or (args.context is not None and args.context.exists()):
        raise FileExistsError("Output/context already exists; use a new attempt")
    start = time.monotonic()
    context = {"status": "running", "start_utc": utc(), "executable": sys.executable,
               "python": sys.version, "cwd": str(Path.cwd()), "argv": sys.orig_argv,
               "stage": args.stage, "output": str(args.output),
               "neural_execution_requested": args.stage != "prepare",
               "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "h5py", "matplotlib", "torch")}}
    if args.context is not None: write_json(args.context, context)
    try:
        case, identity, file_sha = load_case(args.config)
        context.update(config_identity=identity, config_file_sha256=file_sha, requested_device=case["training"]["device"])
        if args.stage == "prepare":
            from .data import prepare
            result = prepare(case, identity, args.output)
        elif args.stage == "train":
            from .training import train
            result = train(case, identity, args.data, args.output, args.smoke)
        elif args.stage == "verify-smoke":
            from .smoke import verify
            result = verify(case, identity, args.data, args.checkpoint, args.output)
        else:
            from .evaluation import analyze
            result = analyze(case, identity, args.data, args.checkpoint, args.output,
                             final=args.stage == "final-evaluate", freeze=getattr(args, "freeze", None))
        context.update(status="completed", exit_code=0)
        print(json.dumps({"stage": args.stage, "status": "completed", "output": str(args.output)}), flush=True)
    except BaseException as error:
        context.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                       error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        context.update(end_utc=utc(), elapsed_seconds=time.monotonic() - start)
        if args.context is not None: write_json(args.context, context)


if __name__ == "__main__":
    main()
