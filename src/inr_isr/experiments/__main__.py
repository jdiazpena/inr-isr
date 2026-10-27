"""User-invoked train/analysis routes through the existing shared3D method."""
import argparse
from pathlib import Path
import shlex
import sys

from .contract import Contract
from inr_isr.synthetic.records import runtime_identity, save_json, utc_now


def train(contract, output, smoke=False, cpu=False):
    from .window import ExperimentWindow
    ExperimentWindow.from_contract(contract).validate_inputs(contract)
    from synthetic_train_3d import build_parser, build_dataset, plot_history, plot_diagnostics, plot_xy_prediction_at_time
    from training_engine import train_window
    args = build_parser().parse_args([])
    for key, value in contract.data["training"].items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown training setting: {key}")
        setattr(args, key, value)
    if contract.data["objective"] == "data_only" and (
        args.lambda_curv_xy != 0 or args.lambda_curv_t != 0 or args.reference_loss_weights):
        raise ValueError("Data-only contract has an active prior")
    args.synthetic_csv = str(contract.file("train"))
    args.output_dir = str(output)
    args.cpu = cpu
    if smoke:
        args.num_steps = 3
    output.mkdir(parents=True, exist_ok=False)
    train_window(args, dataset_factory=build_dataset, plot_history_fn=plot_history,
                 plot_diagnostics_fn=plot_diagnostics, plot_prediction_fn=plot_xy_prediction_at_time,
                 diagnostics_without_regularization=True, experiment=contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "analyze"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU choice; CUDA is the default")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh destination")
    if args.action == "analyze" and args.run is None:
        parser.error("analyze requires --run")
    record = {"started_utc": utc_now(), "status": "running", "action": args.action,
              "command": shlex.join([sys.executable, "-m", "inr_isr.experiments", *sys.argv[1:]]),
              "runtime": runtime_identity()}
    try:
        contract = Contract.load(args.contract)
        if args.action == "train":
            train(contract, args.output, args.smoke, args.cpu)
        else:
            from .analysis import analyze
            analyze(contract, args.run, args.output, cpu=args.cpu)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_utc"] = utc_now()
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / "stage_receipt.json", record)


if __name__ == "__main__":
    main()
