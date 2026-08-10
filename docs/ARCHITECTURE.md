Created: 2026-08-10
Last edited: 2026-08-10
Status: Active architecture contract

# Architecture

## Scientific boundary

The implemented system is a 4D `(x, y, z, t)` neural-field expansion plus
leakage-safe uncertainty evaluation. The currently implemented derivative losses are
optional curvature/smoothness priors. No plasma transport, particle precipitation,
or auroral-image constraint is implemented here. Those remain proposed future
FONDECYT developments and must never be inferred from an extension hook or module
name.

## Preserved baseline

The original copied 3D modules, scripts, configurations, tests, and runbooks remain
the compatibility baseline. The additive implementation lives under
`src/inr_isr_4d`; the working 3D engine has not been converted into a generalized
dimension-independent framework.

## Current additive layers

1. `model.py` specializes the copied SIREN at exactly four inputs.
2. `data.py` separates an immutable full-field bundle from conventional samples.
3. `config.py` defines one strict, serializable, validated 4D contract.
4. `collocation.py` constructs exact independent derivative-evaluation pools.
5. `regularization.py` implements accurately named derivative priors and explicit
   normalized-versus-physical chain-rule scaling.
6. `controller.py` optionally adapts horizontal, vertical, and temporal weights.
7. `training.py` is the one canonical 4D training path used by all later synthetic
   and PFISR adapters.
8. `checkpoint.py` atomically saves complete resumable state.

Conformal calibration will consume a trained checkpoint, saved preprocessing, and a
saved split in a later evaluation stage. It will not be another training program.

## Batching and memory contract

Observation minibatch size, collocation-pool size, collocation batch size, derivative
microbatch size, inference chunk size, and fixed diagnostic-probe size are distinct
configuration values. Derivative microbatches are reduced by their exact fraction of
the collocation batch, so their gradient and scalar reduction match an unchunked
mean. This supports logical collocation pools larger than immediate 16 GB GPU memory.

## Output safety contract

Run directories are collision-refusing and are never silently overwritten. A
checkpoint is atomically replaced and contains model, optimizer, step, best metric,
preprocessing, exact split state, controller state, resolved configuration, and
Python/NumPy/PyTorch CPU/CUDA RNG states. A completion marker is written only after
the configured final step. Explicit restart still requires a new output directory so
historical results remain intact.
