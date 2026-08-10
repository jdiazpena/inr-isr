Created: 2026-08-10
Last edited: 2026-08-10
Status: Active implementation provenance

# Implementation provenance

## Scope boundary

This repository implements and evaluates a 4D expansion of the working UAI-stage
neural field and leakage-safe conformal prediction. Its derivative losses are
curvature or smoothness priors. It does not implement plasma-transport equations,
precipitation models, auroral-camera assimilation, or the complete physics-informed
system proposed as future FONDECYT work.

## Copied foundation

The repository's initial commit is a literal copy of the complete working tree at
`/home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d`, subject only to the exclusions
recorded in `docs/BASELINE_PROVENANCE.md`. The source Git history was not copied.
The copied implementation itself, not a recreation or compatibility adapter, is the
3D baseline.

## Additive component map

| Target component | Classification | Exact inspected reference | Known limitation retained or avoided | Independent acceptance evidence | Reason |
|---|---|---|---|---|---|
| `inr_isr_4d.model.SIREN4D` | New direct specialization of copied foundation | Copied `src/models.py::MLPINR`; comparison reference `inr-isr/src/models.py::SIREN4D` | Reference wrapper added no new mathematics; target likewise adds none and fixes input size at four | Exact identical-weight output comparison; shape and first/second gradient tests | Preserve proven SIREN initialization and forward behavior without copying a 4D training architecture |
| `legacy_diagonal_4d` | Clean implementation of an individually inspected equation | `inr-isr/src/training_common.py::curvature_loss_4d`, isotropic branch | It omits mixed Hessian terms, so it is never called isotropic in the target | Analytic diagonal/mixed derivative tests and exact scalar loss | Retain a precise legacy comparison path |
| `anisotropic_huber_4d` | Clean implementation of an individually inspected equation | `inr-isr/src/training_common.py::curvature_loss_4d`, anisotropic branch | It is a derivative prior, not a physical equation; space-time mixed terms remain absent | Analytic full-component and Huber scalar tests | Retain the existing horizontal-Frobenius/vertical-Huber/temporal formulation for bounded comparison |
| `spatial_hessian_3d` | New implementation | Equation re-derived from the spatial Hessian Frobenius norm; existing reference supplied only the x-y mixed subset | Adds all spatial mixed terms with multiplicity two; excludes space-time mixed terms | Analytic x-y, x-z, y-z and total-norm tests | Provide an accurately named optional full-spatial comparator |
| `FieldBundle4D` and `SampleDataset4D` | New implementation | Ambiguous behavior inspected in `inr-isr/src/amisr_h5_reader_4d.py::AMISR4DVolumeDataset` | Avoids reference behavior where `len` reported rows but item zero returned the full field | Conventional indexing, length, bounds, immutability, and train-only-scaler tests | Separate immutable full-field access from conventional sample batching |
| Strict 4D configuration | New implementation | Failure modes documented in audit; legacy CLI/config code inspected | Rejects rather than ignores unknown or incompatible controls; explicit zero overrides survive | Unknown-field, invalid-combination, round-trip, and `seed=0` tests | Prevent declared knobs from becoming inert |
| Collocation pools | New implementation | Existing trainers inspected; their derivative set was the training coordinates and `num_collocation` was inert | Support-aware mode is geometric proximity sampling, not a claim of complete radar observability | Exact count, deterministic seed, mode, pool-size, and batch-size tests | Make derivative evaluation independent and operational |
| Canonical 4D trainer | New implementation following the copied 3D operational pattern | Copied `src/training_engine.py`, `src/training_common.py`, and trainer entry points | Adds a distinct 4D path; it does not dimension-generalize or modify the copied 3D engine | Independent data/collocation batch tests, microbatch reduction equivalence, structured history, collision refusal, and bounded optimization | Carry forward proven operational concepts without inheriting the standalone legacy 4D loops |
| Reference-ratio controller | New generalized implementation | Copied 3D adaptive-weight concepts in `src/training_engine.py`; legacy 4D trainers had no equivalent complete controller | Controls horizontal, vertical, and temporal derivative-prior components only; it supplies no physical closure | Hand-calculated ratio update and exact state round-trip | Preserve a useful 3D control while making every 4D component explicit |
| Complete checkpoint/resume | New implementation | Bare-model checkpoints in the inspected 4D trainers were characterized but not reused | Target state includes optimizer, preprocessing, splits, controller, resolved config, and all required RNG state | Interrupted-plus-resumed training is bitwise identical to the uninterrupted deterministic run | Make long runs genuinely resumable rather than merely reloadable |
| Group splitting | New implementation after behavioral inspection | `inr-isr/src/conformal_prediction_4d.py` beam splitting and tests | Avoids radius-based clustered selection that can silently change requested withheld counts; adds time and joint units plus a distinct validation role | Exact-count, determinism, full partition, and direct group-leakage tests for all units/strategies | Make withholding reproducible and prevent shared-group leakage |
| Finite-sample conformal core | Clean implementation of an inspected score convention with corrected quantile selection | `inr-isr/src/conformal_prediction_4d.py` and `tests/test_conformal_prediction_4d.py` | Retains absolute-residual symmetric intervals; replaces default interpolated `np.quantile` and rejects invalid empty/non-finite inputs | Hand-calculated upper-rank arrays, invalid-input tests, interval/coverage tests | Preserve the bounded useful convention while implementing valid finite-sample selection |
| Checkpoint-consuming UQ stage | New implementation | Legacy conformal trainer/evaluator orchestration inspected, not copied | Calibration cannot train or tune the model; checkpoint split and preprocessing must match exactly | End-to-end bounded train/calibrate/test test, model SHA-256, stratification, bootstrap, and output-schema checks | Enforce leakage-safe separation and auditable empirical reporting |

No standalone 4D trainer, configuration tree, output directory, or duplicated package
layout from `inr-isr` has been copied.

## Legacy behaviors retained for explicit comparison

- Coordinate order: normalized `(x, y, z, t)`.
- Model output: one normalized log10 electron-density value per coordinate.
- `legacy_diagonal_4d`: mean squared `f_xx`, `f_yy`, `f_zz`, and `f_tt`.
- `anisotropic_huber_4d`: full x-y Hessian Frobenius term, vertical Huber
  curvature, and squared temporal curvature with independently configurable weights.
- Space-time mixed derivatives are not part of any default.
- Existing synthetic alternatives and beam/conformal conventions will be retained as
  named comparisons only after their later component-level characterization.
