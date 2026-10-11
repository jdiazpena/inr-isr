"""Generation, shorter windows and truth rendering only; no training commands."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

import pandas as pd

from .records import file_identity, runtime_identity, save_json, utc_now
from .tables import observation_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "stills", "render"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--integration-sec", type=float, default=0)
    parser.add_argument("--integration-spacing-sec", type=float, default=0.1)
    parser.add_argument("--times-sec", type=float, nargs="+")
    parser.add_argument("--frame-step-sec", type=float, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=192)
    args = parser.parse_args()
    definition = json.loads(args.recipe.read_text())
    geometry = pd.read_csv(args.geometry)
    if args.action == "render":
        for name in ("ffmpeg", "ffprobe"):
            if shutil.which(name) is None:
                parser.error(f"{name} must be installed in the Docker system before rendering")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"started_utc": utc_now(), "status": "running", "action": args.action,
               "command": shlex.join([sys.executable, "-m", "inr_isr.synthetic", *sys.argv[1:]]),
               "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "objective": "Generate/review frozen analytic synthetic data; no neural work",
               "inputs": [file_identity(args.recipe), file_identity(args.geometry)],
               "runtime": runtime_identity(), "outputs": []}
    receipt_path = args.output / "receipt.json"
    save_json(receipt_path, receipt)
    try:
        save_json(args.output / "scene_definition.json", definition)
        geometry.to_csv(args.output / "geometry.csv", index=False)
        if args.action == "generate":
            for case in definition["cases"]:
                dest = args.output / case["id"]
                dest.mkdir()
                df = observation_table(definition, geometry, case["speed_km_s"],
                                       start_sec=args.start_sec, end_sec=args.end_sec,
                                       exposure_sec=args.integration_sec,
                                       integration_spacing_sec=args.integration_spacing_sec)
                path = dest / "synthetic_observations.csv"
                df.to_csv(path, index=False)
                receipt["outputs"].append({"case": case, "file": file_identity(path),
                    "rows": len(df), "records": int(df.time_index.nunique()),
                    "time_bounds_sec": [float(df.t_sec.min()), float(df.t_sec.max())],
                    "ne_bounds_m3": [float(df.Ne.min()), float(df.Ne.max())]})
                print(f"Generated {case['id']}: {len(df):,} point samples, {df.time_index.nunique():,} records", flush=True)
        elif args.action == "stills":
            from .render import write_stills
            times = args.times_sec or [0, definition["duration_sec"] / 2, definition["duration_sec"]]
            receipt["outputs"] = write_stills(definition, geometry, args.output, times, grid_size=args.grid_size)
        else:
            from .render import write_movie
            result = write_movie(definition, geometry, args.output, start_sec=args.start_sec,
                                 end_sec=definition["duration_sec"] if args.end_sec is None else args.end_sec,
                                 frame_step_sec=args.frame_step_sec, fps=args.fps, grid_size=args.grid_size)
            receipt["outputs"] = [result]
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["ended_utc"] = utc_now()
        save_json(receipt_path, receipt)
    print(f"{args.action} complete: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
