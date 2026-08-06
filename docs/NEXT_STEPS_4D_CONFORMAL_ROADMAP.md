# Roadmap: 4D Field Expansion & Held-Out Beam Conformal Calibration (`inr-isr`)

**Target Package:** `inr-isr`  
**Purpose:** Instructions for an autonomous AI agent (or developer) to implement the next high-impact baseline features for the UAI Postdoc codebase prior to grant evaluation.  
**Created:** 2026-08-06  
**Status:** Ready for autonomous agent execution via `/goal`  

---

## Executive Objective

Extend the existing `inr-isr` baseline software from 3D `(x, y, t)` single-altitude slice reconstruction to:
1. **Full 4D Coordinate Representation `(x, y, z, t) -> log10 Ne`:** Continuous space-time volume interpolation across altitude range gates ($100 \le z_{\text{km}} \le 500$).
2. **Held-Out Radar Beam Conformal Calibration (`C_{0.95}`):** Distribution-free 95% prediction interval calibration using 15% withheld radar beams across synthetic patch data and real PFISR radar data.

---

## Task Execution Breakdown

### Phase 1: Full 4D Coordinate Expansion `(x, y, z, t)`

1. **Update Data Normalization (`src/inr_radar/datasets/coordinate_transforms.py`):**
   - Extend `Normalizer3D` to `Normalizer4D`, scaling continuous space-time coordinates $(x_{\text{km}}, y_{\text{km}}, z_{\text{km}}, t_{\text{sec}})$ to normalized hypercube $[-1, 1]^4$.
   - Maintain inverse coordinate transforms for visualization: $x = \text{unnormalize}(x_{\text{norm}})$.

2. **Update Multi-Gate Data Ingestion (`src/inr_radar/datasets/amisr_h5.py`):**
   - Update `read_amisr_hdf5_3d` / `read_amisr_hdf5_4d` to extract multiple altitude gates ($z_{\text{km}}$ from 100 to 500 km) instead of filtering to a single altitude slice.
   - Return DataFrame containing `[x_km, y_km, z_km, t_sec, log10_Ne, dNe, beam_index, beamcode]`.

3. **Update 4D SIREN Coordinate Network (`src/inr_radar/models/siren.py`):**
   - Set `in_features = 4` as default for `MLPINR` (`x_norm, y_norm, z_norm, t_norm`).
   - Verify PyTorch Autograd spatial gradients ($\nabla \hat{N}_e = [\frac{\partial \hat{N}_e}{\partial x}, \frac{\partial \hat{N}_e}{\partial y}, \frac{\partial \hat{N}_e}{\partial z}]$) and temporal derivative ($\frac{\partial \hat{N}_e}{\partial t}$).

---

### Phase 2: Held-Out Radar Beam Split Conformal Prediction (`C_{0.95}`)

1. **Implement Conformal Calibration Module (`src/inr_radar/uq/conformal.py`):**
   - Create `SplitConformalCalibrator`:
     - **Beam Splitting:** Randomly select 15% of radar beams (e.g. 7 beams out of 48 for PFISR, or 6 beams out of 42 for synthetic data) as the calibration set $\mathcal{D}_{\text{cal}}$. Train the 4D SIREN network on the remaining 85% beams $\mathcal{D}_{\text{train}}$.
     - **Non-Conformity Residuals:** On calibration set $\mathcal{D}_{\text{cal}}$, compute non-conformity scores:
       $$s_i = \left| y_i - \log_{10} \hat{N}_e(\mathbf{x}_i) \right|$$
     - **Empirical Quantile Calculation:** Compute the 95th percentile conformal quantile:
       $$q_{0.95} = \text{Quantile} \left( \{s_i\}_{i \in \mathcal{D}_{\text{cal}}}, \frac{\lceil(N_{\text{cal}}+1) \times 0.95\rceil}{N_{\text{cal}}} \right)$$
     - **Prediction Intervals:** Form distribution-free 95% prediction intervals at arbitrary space-time coordinates:
       $$\mathcal{C}_{0.95}(\mathbf{x}) = \left[ \log_{10} \hat{N}_e(\mathbf{x}) - q_{0.95}, \; \log_{10} \hat{N}_e(\mathbf{x}) + q_{0.95} \right]$$
     - **Empirical Coverage Verification:** Calculate empirical coverage fraction:
       $$\text{Coverage} = \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{i \in \mathcal{D}_{\text{cal}}} \mathbb{I} \left( y_i \in \mathcal{C}_{0.95}(\mathbf{x}_i) \right)$$
       Verify that $\text{Coverage} \ge 0.95$.

---

### Phase 3: Benchmark Drivers & Publication Figures

1. **Synthetic 4D Conformal Benchmark (`benchmarks/run_4d_conformal_synthetic.py`):**
   - Generate synthetic 3D space + 1D time moving patch dataset ($v = 1.0 \text{ km/s}$).
   - Train 4D SIREN with 15% held-out beams.
   - Save coverage metrics and plots to `outputs/synthetic_4d_conformal/`.

2. **Real PFISR 4D Conformal Benchmark (`benchmarks/run_4d_conformal_pfisr.py`):**
   - Load real PFISR event dataset across 100 to 500 km altitudes.
   - Withhold 7 real PFISR beamcodes.
   - Train 4D SIREN and compute conformal intervals $\mathcal{C}_{0.95}$ on the 7 held-out real beams.
   - Render 95% confidence band plots comparing predictions $\hat{N}_e$ vs true measurements $y_i$ along held-out beams.

3. **Update `README.md` & Commit:**
   - Add the 4D held-out beam conformal plot to `README.md`.
   - Run unit tests (`python3 -m unittest discover tests/`) to ensure all 13+ tests pass cleanly.
   - Commit and push changes to `https://github.com/jdiazpena/inr-isr`.

---

## Verification Criteria

An AI agent executing this goal should verify completion by ensuring:
- [ ] `python3 -m unittest discover tests/` runs and passes 100% of unit tests.
- [ ] 4D SIREN accepts inputs of shape `[N, 4]` and outputs predictions `[N, 1]`.
- [ ] Conformal coverage on held-out beams achieves $\ge 95\%$ empirical coverage.
- [ ] `git status` is clean after committing changes.
