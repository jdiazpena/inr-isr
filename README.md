# `inr-isr`: Baseline Implicit Neural Field Codebase for ISR Plasma Field Reconstruction

**Applicant:** Joaquín Mateo Díaz Peña  
**Institutional Pre-Submission Baseline Repository for ANID FONDECYT Postdoctorado 2027**  
**Evaluation Panel:** Ingeniería 2 (Electrical Engineering, Computing, Artificial Intelligence & Signal Processing)  
**Project Title:** *Physics-Informed Neural Fields for 4D Ionospheric Reconstruction: Integrating Multi-Beam Radar, Plasma Transport, and Auroral Imaging*

---

## 1. Executive Summary & Purpose

`inr-isr` is an open-source Python codebase containing the **verified baseline software developed during the applicant's UAI Postdoctoral fellowship**. It provides empirical proof of completed software capabilities prior to proposal submission, implementing Implicit Neural Representations (INRs)—specifically SIREN-style periodic activation networks—to reconstruct continuous 4D ionospheric plasma fields from sparse remote sensing observations.

This repository serves as direct evidence for ANID FONDECYT Postdoctorado 2027 evaluators that the applicant has already implemented and validated:
1. **Production-Grade Data Readers:** Ingesting native AMISR database HDF5 fit files from 48-beam AMISR phased-array radars (PFISR, Alaska and RISR-N, Resolute Bay) as well as auxiliary Madrigal HDF5 products.
2. **Differentiable Coordinate Architectures:** 3-layer SIREN MLPs with exact PyTorch Automatic Differentiation (`torch.autograd`).
3. **Synthetic OSSE Prototyping Engine:** Parameterized 3D/4D moving plasma patch generators for evaluating velocity convection (0.36 to 3.00 km/s) across sparse radar beam geometries (42, 23, and 11 beams) and integration times (1 to 10 min).
4. **Automated Regularization Telemetry:** Horizontal Hessian curvature penalty loops ($\mathcal{L}_{xy} = f_{xx}^2 + 2f_{xy}^2 + f_{yy}^2$) and loss ratio diagnostic tracking.

---

## 2. Demonstrated Baseline Capabilities (Completed UAI Work)

```text
+-----------------------------------------------------------------------------------+
|                  VERIFIED BASELINE RESULTS (36 EXPERIMENTAL RUNS)                 |
+-----------------------------------------------------------------------------------+
| 1. Real Radar Ingestion: Production-grade reading of PFISR/RISR native AMISR      |
|    database HDF5 fit files (Ne, Te, Ti, v_los, measurement variances).            |
| 2. Measured-Field Improvement: Hessian curvature regularization improves active-  |
|    region RMSE across 100% of synthetic pilot test cases.                         |
| 3. Motion Blur Mismatch: Demonstrated that fast convection (2.0 km/s) over 10-min  |
|    integration produces anisotropic motion blur (Sigma_eff = Sigma_0 + T_int^2/12)|
|    proving the necessity of flow-aware velocity operators.                        |
+-----------------------------------------------------------------------------------+
```

---

## 3. UAI Baseline Architecture & Optimization Topology

The baseline model maps normalized space-time coordinates $X = (x, y, z, t) \in [-1, 1]^4$ to continuous plasma parameter predictions $\log_{10} \hat{N}_e(x,y,z,t)$.

The optimization objective balances heteroscedastic radar measurement noise with baseline derivative penalties:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda_{xy} \mathcal{L}_{xy}(\theta) + \lambda_t \mathcal{L}_t(\theta)$$

Where:
- **Radar Observation Loss:** $\mathcal{L}_{\text{data}}(\theta) = \frac{1}{N_{\text{obs}}} \sum_{i=1}^{N_{\text{obs}}} \frac{|y_i - \log_{10} \hat{N}_e(\mathbf{x}_i)|^2}{\sigma_{\text{radar}}^2(\mathbf{x}_i)}$
- **Horizontal Hessian Curvature Penalty:** $\mathcal{L}_{xy}(\theta) = \mathbb{E}_{\Omega} \left[ f_{xx}^2 + 2 f_{xy}^2 + f_{yy}^2 \right]$
- **Temporal Bending Penalty:** $\mathcal{L}_t(\theta) = \mathbb{E}_{\Omega} \left[ f_{tt}^2 \right]$

---

## 4. Proposed FONDECYT Postdoctorado 2027 Project Delta

The proposed 36-month FONDECYT fellowship will extend this proven software baseline into a full **Physics-Informed Neural Field (PI-INR)** framework by:
- **Replacing Ad-hoc Curvature Loss with 4D Plasma Continuity PDE ($\mathcal{L}_{\text{PDE}}$):** Coupling continuous PyTorch autograd derivatives directly to derived 3D plasma drift vectors ($\mathbf{v}_{\text{Semeter}}$) and multi-spectral auroral electron production ($q_{\text{aurora}}$).
- **Benchmarking on 3D GEMINI OSSE Simulations:** Validating inter-beam advection against full 1.2 TB multi-fluid storm simulations.
- **Conformal Uncertainty Quantification ($C_{0.95}$):** Calibrating distribution-free 95% coverage prediction intervals on held-out radar beams.

---

## 5. Quick Start & Execution

```bash
# Clone and install the baseline package
git clone https://github.com/jdiazpena/inr-isr.git
cd inr-isr
pip install -e .

# Run unit tests to verify installation
python3 -m unittest discover tests/

# Run the synthetic velocity & beam support benchmark
python3 benchmarks/run_synthetic_benchmark.py --config configs/synthetic_patch_config.json
```

---

## 6. Citation & Proposal Reference

```bibtex
@misc{DiazPena2026INRISRBaseline,
  author = {D{\'\i}az Pe{\~n}a, Joaqu{\'\i}n Mateo},
  title = {inr-isr: Baseline Implicit Neural Field Codebase for ISR Field Reconstruction},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jdiazpena/inr-isr}}
}
```
