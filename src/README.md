# Scientific Source

This folder contains the implementation that performs data loading, field generation,
model construction, training, and reconstruction analysis.

| File | Responsibility |
|---|---|
| `models.py` | Neural-network components and the `MLPINR` model |
| `amisr_h5_reader_3d.py` | Reads AMISR HDF5 products into a physical-coordinate dataframe |
| `datasets.py` | Real radar HDF5 dataset loading and normalization |
| `synthetic_dataset.py` | Synthetic observation dataset and normalization |
| `synthetic_plasma.py` | Synthetic plasma fields, motion, beam geometries, and observation generation |
| `synthetic_train_3d.py` | Current synthetic trainer with derivative diagnostics and adaptive regularization |
| `synthetic_train_3d_window_reference_reg.py` | Earlier/reference synthetic training implementation |
| `train_radar_3d_window_reg.py` | Real-radar window trainer |
| `train_radar_3d_window_reference_reg_diagnostic.py` | Real-radar trainer with reference-ratio diagnostics |
| `synthetic_analyze_reconstruction.py` | Accepted reconstruction analysis in linear density units |
| `synthetic_analyze_reconstruction_log_errors.py` | Older log-space analysis retained for comparison |
| `synthetic_analyze_reconsturction*.py` | Compatibility aliases for historical misspelled commands |
| `training_common.py` | Shared collocation, curvature, diagnostics, and evaluation utilities |
| `training_engine.py` | Shared active optimization loop used by synthetic and real-radar SIRENs |
| `training_config.py` | Typed SIREN, optimization, regularization, and diagnostic defaults |

The active training path is intentionally layered: a data-specific adapter calls the
shared engine, which calls the common numerical utilities. Historical trainers remain
separate to preserve their earlier output schemas. See `docs/SCIENTIFIC_PIPELINE.md`
and `docs/REFACTORING_NOTES.md` for equations and equivalence evidence.
