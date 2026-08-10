# Context for new chat: SIREN / INR with real radar data

## Who I am and what I am trying to do

I am working on a scientific machine learning project using implicit neural representations (INRs), especially SIREN-like neural fields, for sparse radar-derived ionospheric data.

The long-term project is to reconstruct continuous 4D ionospheric fields from sparse multi-beam incoherent scatter radar (ISR) observations. The target variables are ISR-derived plasma parameters such as:

- Electron density, Ne
- Electron temperature, Te
- Ion temperature, Ti

The full model will eventually be a continuous neural field:

\[
f_\theta(x,y,z,t) \rightarrow [N_e, T_e, T_i]
\]

But I want to start simpler, probably with real radar data in 2D first, before moving to 3D + time.

The immediate goal is not to build a perfect final method. The immediate goal is to learn how to use SIREN / neural fields on real radar data and understand the mechanics: coordinate normalization, sparse sampling, uncertainty-weighted losses, interpolation behavior, artifacts, and validation.

---

## Important clarification: this is not full radar inversion

Do not assume I am building a full radar forward model.

I am not trying to simulate raw ISR spectra, autocorrelation functions, radar power, scattering physics, or the full ISR fitting pipeline.

The data I plan to use are already processed ISR-derived products. So the model should start from fitted quantities like Ne, Te, Ti, with their reported uncertainties, beam geometry, range gates, and timestamps.

The correct framing is:

> Sparse field reconstruction using a neural field and an observation/sampling operator.

The minimal observation operator is just point sampling:

\[
\hat{y}_j = H_j[f_\theta] = f_\theta(x_j, y_j, z_j, t_j)
\]

or for a 2D first experiment:

\[
\hat{y}_j = f_\theta(r_j, t_j)
\]

or:

\[
\hat{y}_j = f_\theta(\text{altitude}_j, t_j)
\]

The data loss should initially be:

\[
\mathcal{L}_{data}
=
\sum_j
\frac{(y_j - \hat{y}_j)^2}{\sigma_j^2}
\]

where \(\sigma_j\) is the ISR-reported uncertainty for that fitted parameter.

If no uncertainty is available at first, use ordinary MSE as a temporary learning baseline, but the real project should use heteroscedastic weighting.

A more advanced observation operator could later average over range-gate volumes:

\[
H_j[f_\theta]
=
\int_{\Omega_j} w_j(x,y,z) f_\theta(x,y,z,t_j)dV
\]

but this is not the first implementation.

---

## The planned progression

The practical path should be incremental:

1. Dense toy SIREN fitting, such as images or synthetic fields.
2. Sparse toy fitting, such as sparse image pixels or synthetic beam samples.
3. Real radar data in 2D, probably altitude versus time for one beam or one variable.
4. Real radar data with sparse/irregular sampling across beam/range/time.
5. Multi-output field for Ne, Te, Ti.
6. 3D + time reconstruction.
7. Add uncertainty and reliability products.
8. Compare against interpolation/smoothing baselines.

For the new chat, start with step 3 if I provide radar files.

Do not jump straight to 4D unless the 2D pipeline works.

---

## First real-radar target

A good first real-data problem is:

\[
f_\theta(h,t) \rightarrow \log_{10}(N_e)
\]

where:

- \(h\) is altitude or range-derived altitude
- \(t\) is time
- output is log electron density

Why log Ne?

- Ne often spans orders of magnitude.
- A log transform makes the numerical scale easier for neural fields.
- It makes MSE less dominated by the largest densities.

Then later try:

\[
f_\theta(h,t) \rightarrow [\log_{10}(N_e), T_e, T_i]
\]

or train separate networks per variable first.

---

## Coordinate normalization is critical

For SIREN and ReLU+Fourier baselines, coordinates must be normalized.

Usually normalize every coordinate to \([-1,1]\):

\[
x_{norm} = 2 \frac{x - x_{min}}{x_{max} - x_{min}} - 1
\]

For 2D radar data:

- altitude/range should be normalized to \([-1,1]\)
- time should be normalized to \([-1,1]\)

Do not feed raw Unix time into a SIREN.

Do not feed raw altitude in kilometers without normalization.

The model may fail simply because scaling is bad.

---

## SIREN baseline

Use vanilla SIREN first.

The original SIREN idea:

\[
\Phi(x) = W_n(\phi_{n-1}\circ...\circ\phi_0)(x)+b_n
\]

with:

\[
\phi_i(x_i)=\sin(W_i x_i+b_i)
\]

Key practical points:

- First-layer frequency \(\omega_0\) matters a lot.
- The original paper used \(\omega_0=30\), but that is not automatically correct for radar data.
- Hidden-layer initialization matters.
- SIREN works well when derivatives are needed, but it can hallucinate high-frequency structure under sparse sampling.

For radar, sweep \(\omega_0\):

- 5
- 10
- 15
- 30
- 45
- 60

Do not assume the highest \(\omega_0\) is best. Sparse data may prefer lower bandwidth.

---

## Baselines to compare against

Do not compare only SIREN versus nothing.

At minimum:

1. Interpolation baseline
   - nearest, linear, cubic, or gridded interpolation
   - plus optional smoothing

2. ReLU + Fourier features
   - coordinate encoding followed by a ReLU MLP
   - important sanity check because SIREN is not always superior

3. Vanilla SIREN
   - main learning baseline

Later, if the basic pipeline works:

4. BACON
   - band-limited coordinate network
   - useful because sparse reconstruction needs bandwidth control

5. FINER
   - SIREN-like variable-periodic activation
   - likely useful if vanilla SIREN has poor spectral behavior

6. WIRE
   - localized Gabor/wavelet activations
   - likely useful if SIREN rings, oversmooths, or creates global artifacts

But for the first real radar pass, start with interpolation, ReLU+Fourier, and SIREN.

---

## ReLU + Fourier baseline

There may not be a single canonical repo for ReLU + Fourier because it is a baseline recipe, not one unique model.

The structure is:

\[
x \rightarrow \gamma(x) \rightarrow \text{ReLU MLP} \rightarrow f(x)
\]

where \(\gamma(x)\) is a Fourier feature mapping.

A simple NeRF-style encoding is:

\[
\gamma(x) = [x, \sin(\pi x), \cos(\pi x), \sin(2\pi x), \cos(2\pi x), ...]
\]

Use it as a sanity check. If SIREN cannot beat ReLU+Fourier on the same sparse radar split, that is important information.

---

## Loss functions

Start with data-only:

\[
\mathcal{L} = \mathcal{L}_{data}
\]

Then add temporal smoothness:

\[
\mathcal{L}_{time}=\left\|\frac{\partial f_\theta}{\partial t}\right\|^2
\]

Then add spatial/altitude smoothness:

For 2D altitude-time:

\[
\mathcal{L}_{h}=\left\|\frac{\partial f_\theta}{\partial h}\right\|^2
\]

Full loss:

\[
\mathcal{L}
=
\mathcal{L}_{data}
+
\lambda_t\mathcal{L}_{time}
+
\lambda_h\mathcal{L}_{h}
\]

For 3D + time later:

\[
\mathcal{L}_{smooth}
=
\alpha_x\left\|\partial_x f_\theta\right\|^2
+
\alpha_y\left\|\partial_y f_\theta\right\|^2
+
\alpha_z\left\|\partial_z f_\theta\right\|^2
\]

Eventually, anisotropic smoothness should reflect the physics. But for the first real-data implementation, keep it simple.

Important: do not add priors before the data-only fit works.

---

## Validation strategy

The project must avoid simply making pretty plots.

Use withheld tests:

1. Random held-out samples
   - easiest
   - but may overestimate performance if nearby samples remain in training

2. Withheld time blocks
   - tests temporal interpolation/generalization

3. Withheld altitude/range bands
   - tests vertical interpolation/generalization

4. Withheld beam
   - later, for multi-beam data
   - most relevant to the final project

Metrics:

- MAE
- RMSE
- normalized RMSE
- error by altitude
- error by time
- uncertainty-weighted residuals
- coverage of predictive intervals, if uncertainty is implemented

For Ne, use metrics on log Ne first.

---

## Uncertainty plan

The long-term project wants reliability maps and uncertainty products.

There are three levels:

### Level 1: ISR-reported measurement uncertainty

Use \(\sigma_j\) in the training loss:

\[
\mathcal{L}_{data}
=
\sum_j
\frac{(y_j-\hat{y}_j)^2}{\sigma_j^2}
\]

This is aleatoric/instrument uncertainty.

### Level 2: epistemic uncertainty

Use MC dropout or deep ensembles to estimate uncertainty due to weak data coverage.

MC dropout is easier.

Deep ensembles are more expensive but often more reliable.

### Level 3: conformal calibration

Conformal prediction should be considered later as a calibration wrapper.

Basic conformal score:

\[
s_j = \frac{|y_j-\hat{y}_j|}{\hat{\sigma}_j}
\]

Then use a calibration split to estimate a quantile \(q_{1-\alpha}\), and produce calibrated intervals:

\[
\hat{y} \pm q_{1-\alpha}\hat{\sigma}
\]

For ISR, global conformal prediction may be too crude. Better options later:

- stratify by altitude
- stratify by variable, Ne/Te/Ti
- stratify by beam/elevation
- stratify by SNR or uncertainty bin
- use withheld-beam or withheld-time calibration

Important caveat: conformal guarantees apply to the distribution of calibration/test points. If calibration points are radar samples, the guarantee is for radar-sample predictions, not automatically for every unobserved voxel.

---

## What the previous literature search found

The prior research looked at SIREN and INR work since 2020.

Main conclusion:

> The field did not converge on “SIREN is always best.” SIREN is a strong baseline, especially when derivatives are needed, but later methods often improve spectral control, locality, initialization, or conditioning.

Relevant methods:

### SIREN

Baseline periodic activation INR. Useful because derivatives are clean and accessible. Sensitive to initialization and coordinate scaling.

### FINER

Variable-periodic activation. Strong candidate if we want a low-overhead SIREN-like improvement.

### WIRE

Gabor/wavelet activation. Useful if sparse radar fields show localized structures or if SIREN creates ringing/global artifacts.

### BACON

Band-limited coordinate network. Especially relevant because sparse measurements cannot justify arbitrary high-frequency structure.

### H-SIREN

Small first-layer modification. Possible low-cost experiment, but not a first priority.

### ReLU + Fourier features

Important baseline. A well-tuned ReLU+Fourier model may be competitive and should not be ignored.

### Grids / interpolation

Do not assume neural fields always beat grids. Interpolation and smoothing are required baselines.

---

## What to ignore for now

Do not spend time on these unless directly needed:

- SIREN for image compression
- SIREN for audio
- GAN-based SIREN papers
- pi-GAN and generative graphics
- general NeRF rendering unless it is specifically about dynamic 4D factorization or sparse reconstruction
- huge literature surveys without implementation value

The project is not about making a better image INR. It is about reliable sparse scientific field reconstruction.

---

## The main research framing

The correct research question is not:

> How do I improve SIREN in general?

The correct question is:

> Which neural-field representation and regularization strategy gives reliable reconstruction of sparse, uncertain radar-derived ionospheric fields?

SIREN is one candidate backbone. The actual contribution will likely come from:

- sparse observation modeling
- uncertainty-weighted fitting
- temporal and spatial regularization
- withheld-beam/withheld-time validation
- reliability/observability diagnostics
- careful comparison to interpolation and ReLU+Fourier baselines

---

## First coding target for the new chat

If I provide radar data, help me build this first:

1. Load radar data.
2. Extract one variable, likely Ne.
3. Convert time and altitude/range to normalized coordinates.
4. Optionally use log10(Ne).
5. Split into train/validation/test.
6. Train vanilla SIREN on \((h,t)\rightarrow \log_{10}(N_e)\).
7. Compare against interpolation.
8. Plot:
   - observed data
   - SIREN reconstructed field on a regular grid
   - interpolation baseline
   - error on held-out samples
   - residuals by altitude and time
9. Then add ReLU+Fourier as a baseline.
10. Then add uncertainty-weighted loss if \(\sigma\) is available.

Do not start with a complicated architecture.

---

## Useful implementation details

### Data table format

A convenient internal table should look like:

| time | altitude | beam | variable | value | sigma |
|---|---:|---|---|---:|---:|
| t_j | h_j | beam_id | Ne | y_j | sigma_j |

For a first model, filter to:

- one beam
- one variable
- one time interval

Then build coordinates:

\[
X_j = [h_{norm,j}, t_{norm,j}]
\]

and targets:

\[
y_j = \log_{10}(N_{e,j})
\]

### Data cleaning

Before training:

- remove NaNs
- remove negative or zero Ne before log transform
- remove impossible temperatures
- optionally remove very large uncertainty samples
- keep a mask of what was removed

### Batch strategy

For small 2D datasets, full-batch training may work.

For larger data:

- use random mini-batches of observed points
- separately sample regular-grid points only for derivative regularization

### Regularization sampling

Derivative priors should be evaluated on points sampled across the domain, not only at observed points, otherwise they may not control empty regions.

Start simple:

- sample random \((h,t)\) points uniformly in normalized domain
- compute \(\partial f/\partial h\) and \(\partial f/\partial t\)
- add smoothness penalty

Use small weights first.

---

## Expected failure modes

1. Model fits training points but creates wild structure between points
   - lower \(\omega_0\)
   - add smoothness
   - use BACON or bandwidth-limited model later

2. Model oversmooths everything
   - increase \(\omega_0\)
   - increase width/depth
   - try FINER later

3. Model has ringing artifacts
   - reduce \(\omega_0\)
   - add regularization
   - try WIRE later

4. Model ignores weak-density regions
   - use log transform
   - normalize targets
   - check uncertainty weights

5. Training unstable
   - check coordinate normalization
   - check SIREN initialization
   - lower learning rate
   - reduce \(\omega_0\)

6. Pretty reconstruction but bad held-out error
   - trust the held-out error
   - add baselines
   - do not claim reconstruction works just from visual smoothness

---

## What I want from the new chat

Help me implement this carefully and incrementally.

Do not jump to a huge final architecture.

Prefer:

- small working code
- clear debugging steps
- explicit tensor shapes
- simple plots
- careful train/test splits
- baselines
- checks for NaNs and scaling

The first deliverable should be a working 2D radar INR pipeline using vanilla SIREN.

