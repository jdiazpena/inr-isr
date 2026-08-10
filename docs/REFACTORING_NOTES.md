# Behavior-Preserving Refactoring Record

## Preserved Scientific Contract

The refactor retains the original SIREN-based INR and its tested behavior:

- `MLPINR` with sine hidden activations and the existing SIREN initialization;
- default 256 hidden features and three hidden layers;
- `first_omega_0 = 5` and `hidden_omega_0 = 5`;
- normalized x-y-time input and normalized `log10(Ne)` output;
- measured-point MSE and the existing spatial/temporal curvature equations;
- adaptive reference-ratio controller, lambda limits, random-number ordering,
  optimizer, checkpoint selection, and history schema;
- integration-averaged synthetic truth as the primary reconstruction target.

## Structural Changes

- Common trainer utilities moved to `src/training_common.py`.
- The active synthetic and real-radar optimization loop moved to
  `src/training_engine.py`.
- Each active trainer now supplies only dataset construction and plotting style.
- Shared defaults are typed in `src/training_config.py` and recorded in
  `config/training_defaults.json`.
- Benchmark matrices moved to `config/velocity_integration_benchmark.json`.
- Synthetic truth equations now have one authoritative implementation in
  `src/synthetic_plasma.py`.
- Dense analysis now uses explicit `ReconstructionAnalysisConfig` objects.
- The accepted analysis name is `synthetic_analyze_reconstruction.py`.

## Compatibility

Historical misspelled `reconsturction` modules remain as aliases. Existing shell
launchers and trainer command-line flags remain valid. JSON configuration is optional;
when omitted, argument namespaces and saved run configurations retain their original
shape.

The older fixed-lambda and earlier reference trainers remain available because their
output schemas differ from the active diagnostic engine. They share low-level
mathematics but retain their historical loops for reproducibility.

## Operational Fix

`scripts/run_reference_windows.py` previously targeted the fixed-lambda trainer while
passing unsupported reference-ratio flags. It now targets the diagnostic
reference-ratio radar trainer and passes the HDF5 path explicitly. Tests parse every
generated wrapper command with the target trainer parser.

## Equivalence Evidence

Before decomposition, deterministic four-step synthetic and tiny-HDF5 radar baselines
were captured with active adaptive regularization and derivative diagnostics. After
refactoring:

- both `history.csv` files were byte-identical;
- every final synthetic and real-radar model tensor was exactly equal;
- the expanded benchmark remained 18 cases and 36 sequential model runs;
- dense physical and gradient reconstruction analysis completed on an existing model;
- synthetic generation and real HDF5 dataset smoke tests passed.

Run the maintained verification suite with:

```bash
bash scripts/run_tests.sh
```
