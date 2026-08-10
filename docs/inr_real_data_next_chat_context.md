# Context for Next Chat: Moving from Image INR to Real ISR / Radar-like Data

## 1. Main objective

The goal is to move from the toy 2D grayscale-image implicit neural representation (INR) experiments to real data.

The long-term project is an uncertainty-aware implicit neural field for incoherent scatter radar (ISR) reconstruction, eventually something like:

\[
\Phi_\theta(x,y,z,t) \rightarrow \text{ionospheric variable}
\]

Possible outputs:

```text
Ne
Te
Ti
Vi
or multiple outputs: [Ne, Te, Ti, Vi]
```

The immediate next step is probably **not full 4D yet**. It may be:

```text
2D radar-like slice
3D volume without time
2D + time
or compressed/simplified real radar data
```

The next chat should help build the first real-data INR carefully, step by step.

---

## 2. What has already been learned from the image experiments

We started with a 2D grayscale image INR:

\[
\Phi_\theta(y,x) \rightarrow I(y,x)
\]

The dataset representation was flattened coordinate-value pairs:

```text
coords: [N, 2]
values: [N, 1]
```

For a 512x512 image:

```text
N = 512 * 512 = 262144
coords.shape = [262144, 2]
values.shape = [262144, 1]
```

Coordinates were normalized to `[-1, 1]`.

Image values were normalized to `[-1, 1]` for training.

Metrics such as PSNR, RMSE, MAE, p95, p99, and bias were computed after converting predictions and targets to `[0, 1]`.

Important distinction:

```text
Training loss: MSE on [-1, 1]
Reported image metrics: usually on [0, 1]
```

PSNR conversion:

```text
MSE_[0,1] = MSE_[-1,1] / 4
PSNR = -10 log10(MSE_[0,1])
```

---

## 3. Current best image-INR baseline

The best current baseline is value-only SIREN-style INR.

Architecture:

```text
MLPINR
input dimension: 2 for image, later 3 or 4 for radar
output dimension: 1 initially
hidden_features = 256
hidden_layers = 3
activation = sine
first_omega_0 = 30
hidden_omega_0 = 30
outermost_linear = True
loss = MSE(value prediction, target value)
optimizer = Adam, lr = 1e-4
```

The corrected SIREN configuration is critical:

```text
first_omega_0 = 30
hidden_omega_0 = 30
```

Earlier bad runs used:

```text
first_omega_0 = 30
hidden_omega_0 = 1
```

That was not equivalent to the original SIREN behavior and caused much worse results.

---

## 4. Critical SIREN lesson

SIREN is not just “use sine activation.”

It is:

```text
coordinate input
+ periodic activations
+ correct frequency scale
+ matching initialization
+ useful phase diversity
```

A sine layer behaves like:

```text
sin(omega_0 * (W x + b))
```

`omega_0` controls how fast the sine oscillates. In an INR, the input is the coordinate itself, so this directly controls what spatial/temporal frequencies the network can represent.

Weights affect frequency/orientation of learned features.

Biases act like phase shifts. In sine networks, this matters because phase controls where oscillations are located.

The original SIREN implementation uses high-frequency sine behavior throughout the network, not only in the first layer.

Practical result:

```text
hidden_omega_0 = 30 caused a large jump in image performance.
hidden_omega_0 = 1 made the network look SIREN-like but behave much worse.
```

---

## 5. Current code status / files

The working codebase has these main files:

```text
datasets.py
models.py
train.py
sweep_sine_omega.py
```

### datasets.py

For image work, it defines:

```text
ImageINRDataset
get_mgrid
load_builtin_camera
```

It creates flattened coordinate-value pairs.

For real radar data, this file will need a new dataset class, probably something like:

```text
RadarINRDataset
```

or separate scripts for loading and preprocessing radar products into:

```text
coords: [N, D]
values: [N, K]
sigma: [N, K] or variance: [N, K]
metadata / masks / quality flags
```

### models.py

Defines:

```text
Sine
MLPINR
activation handling
SIREN initialization
```

For real radar data, the same `MLPINR` can be reused by changing:

```text
in_features = 3 or 4
out_features = 1 or more
```

Examples:

```text
2D slice: in_features = 2
3D volume: in_features = 3
4D volume + time: in_features = 4
single output: out_features = 1
multi-output: out_features = number of physical variables
```

### train.py

Current image training script:

- value-only MSE
- saves model
- saves reconstructions
- saves `history.csv`
- metrics include:
  - raw training MSE on `[-1,1]`
  - raw full MSE on `[-1,1]`
  - full MSE on `[0,1]`
  - PSNR
  - RMSE
  - MAE
  - max absolute error
  - p95 absolute error
  - p99 absolute error
  - bias

### sweep_sine_omega.py

Used for sine-frequency sensitivity analysis.

It sweeps:

```text
first_omega_0
hidden_omega_0
```

It supports manual worker splitting:

```text
worker_id = 0, 1, 2, ...
num_workers = total number of terminals/processes
```

Important: `num_workers` does not automatically spawn processes. The user launches separate terminals manually.

Example for three workers:

```text
terminal 1: --worker_id 0 --num_workers 3
terminal 2: --worker_id 1 --num_workers 3
terminal 3: --worker_id 2 --num_workers 3
```

The sweep saves per-combo histories and per-worker summaries.

---

## 6. Current omega ablation plan

The next analysis on the image side is a sensitivity/ablation study over sine frequencies.

Main question:

```text
How sensitive is image INR performance to first_omega_0 and hidden_omega_0?
```

Suggested staged strategy:

### Stage 1: one-dimensional sweeps

Fix hidden omega and sweep first omega:

```text
first_omega_0 = [5, 10, 15, 20, 30, 45, 60, 90]
hidden_omega_0 = 30
```

Fix first omega and sweep hidden omega:

```text
first_omega_0 = 30
hidden_omega_0 = [5, 10, 15, 20, 30, 45, 60, 90]
```

### Stage 2: local 2D grid

Example:

```text
first_omega_0  = [15, 20, 30, 45, 60]
hidden_omega_0 = [15, 20, 30, 45, 60]
```

This gives 25 combinations.

### Stage 3: rerun best candidates

Take the best 3 to 5 candidates and rerun longer and/or with several seeds.

Important: do not over-interpret one seed.

---

## 7. Metrics to preserve going forward

For images:

```text
MSE
RMSE
MAE
PSNR
max absolute error
p95 absolute error
p99 absolute error
bias
```

For real radar / ISR, PSNR is not meaningful unless the data are image-normalized for a toy diagnostic.

For real physical variables, use:

```text
RMSE
MAE
bias
p95 absolute error
p99 absolute error
correlation
R^2
```

Once uncertainties are included:

```text
normalized residuals
reduced chi-square-like metric
negative log likelihood
coverage of prediction intervals
calibration curves
CRPS maybe later
```

Most important ISR-style normalized residual:

\[
z_i = \frac{\hat{y}_i - y_i}{\sigma_i}
\]

Then check:

\[
\frac{1}{N}\sum_i z_i^2
\]

A value near 1 can indicate statistical consistency if the uncertainties are well calibrated and the model assumptions are appropriate.

---

## 8. Important ISR modeling principle

For ISR, derivative terms should be treated as **soft priors on the learned field**, not target labels computed from sparse radar data.

The planned ISR-style loss is conceptually:

\[
\mathcal{L}
=
\mathcal{L}_{data,\sigma}
+
\lambda_t \mathcal{L}_{continuity}
+
\lambda_s \mathcal{L}_{smoothness}
\]

where:

```text
L_data,sigma = heteroscedastic data loss using radar-provided uncertainties
L_continuity = temporal derivative regularization
L_smoothness = spatial derivative regularization, likely anisotropic
```

At measured radar points:

```text
data loss + optional regularization losses
```

At no-data collocation points:

```text
only regularization losses
```

This means the model must be evaluated at more than just observed data points if regularization is used.

---

## 9. Important distinction: time as depth vs time as coordinate

The next chat should be careful with this.

There was a discussion with Gonzalo Ruz about treating time as “depth.”

The precise distinction:

### CNN / gridded tensor view

Time as depth means data are arranged like:

```text
[T, Z, Y, X]
```

or for video:

```text
[T, Y, X]
```

This is natural for CNNs, U-Nets, or 3D convolutions, but it requires a dense gridded tensor.

### INR / coordinate-based view

In a coordinate-based INR, data are flattened into coordinate-value samples:

```text
coords = [x, y, z, t]
value  = target variable
```

so:

```text
coords.shape = [N, 4]
values.shape = [N, 1] or [N, K]
```

The model is:

\[
\Phi_\theta(x,y,z,t) \rightarrow y
\]

For sparse ISR data, this coordinate-based representation is more natural because the data are not a dense regular 4D cube.

Defensible sentence:

```text
If “time as depth” means treating time as an additional coordinate, then yes, this matches the INR formulation. If it means forcing the ISR data into a dense tensor where time is a depth axis, that is a different gridded CNN-style formulation and may require interpolation or binning before the model.
```

Important citation to remember:

SIREN formulates data as coordinate-value tuples:

\[
D = \{(x_i, a_i(x_i))\}_i
\]

and the neural field as:

\[
\Phi: x \mapsto \Phi(x)
\]

This supports the coordinate-value interpretation.

---

## 10. Gradient and Laplacian lessons

Gradient and Laplacian were implemented for the image case.

Key PyTorch idea:

```text
requires_grad_(True) on coordinates makes input derivatives possible.
First autograd.grad gives first derivatives.
Second autograd.grad on first derivatives gives second derivatives.
```

For image experiments:

- target gradients/laplacians were computed from the known image by finite differences
- model gradients/laplacians were computed by autograd

But for ISR:

- do not compute derivative labels from sparse noisy radar points by finite differences unless there is a very specific reason
- instead, penalize derivatives of the neural field itself as soft priors

Important conclusion:

```text
Gradient and Laplacian losses did not improve dense image PSNR enough to justify using them there.
For ISR, first-derivative priors are still relevant because the problem is sparse, irregular, and uncertainty-weighted.
```

Laplacian should **not** be default for ISR.

Reason:

```text
Laplacian is second-order, expensive, numerically harsh, and can oversmooth/overconstrain physical structure.
```

Keep it only as a later ablation.

---

## 11. Why image interpolation results did not invalidate the ISR project

We tested training at 256x256 and evaluating at 512x512.

Result:

```text
INR 256 -> 512: around 28.65 dB
bicubic 256 -> 512: around 29.05 dB
```

The INR did not beat bicubic in that super-resolution-like test.

Conclusion:

```text
A continuous INR does not magically recover information that was not present in the training data.
```

This does **not** invalidate the ISR project because the ISR project is not about image super-resolution.

ISR value proposition:

```text
sparse irregular sampling
heteroscedastic uncertainties
beam/time geometry
soft priors
reliability/uncertainty products
```

The INR should be benchmarked against interpolation in the real sparse/irregular regime, not dense image upsampling.

---

## 12. Moving to real data: what the next chat should do first

Do not jump immediately to full 4D.

Recommended next steps:

### Step 1: inspect real data structure

Ask the user to provide or describe:

```text
file format: HDF5, NetCDF, MATLAB, CSV, etc.
available variables: Ne, Te, Ti, Vi, errors, SNR, range, beam, az/el, time
coordinate system: beam/range or geographic/cartesian/magnetic coordinates
uncertainty variables: dNe, dTe, dTi, dVi or variances
quality flags / masks
missing values
units
```

### Step 2: build a real-data dataset class

Target structure:

```python
coords: torch.Tensor    # [N, D]
values: torch.Tensor    # [N, K]
sigma: torch.Tensor     # [N, K] or variance [N, K]
mask: optional
metadata: optional
```

Possible first version:

```text
D = 2 or 3
K = 1
loss = ordinary MSE first
```

Only after this works:

```text
add variance-weighted MSE
add uncertainty metrics
add derivative priors
```

### Step 3: choose a first real-data target

Best first target should be simple:

```text
one variable only
one event or time window
one radar mode / experiment
reduced dimensionality
clean quality filtering
```

Possible options:

```text
Ne as log10(Ne)
Te normalized
Ti normalized
one altitude-time slice
one beam range-time slice
one 2D spatial slice
one 3D volume at one time
```

Recommended first physical target:

```text
log10(Ne)
```

Reason:

```text
Ne spans orders of magnitude, so raw Ne is usually poorly scaled for MSE.
```

### Step 4: normalize coordinates and values carefully

Coordinates should be normalized to approximately `[-1, 1]` per dimension.

Example:

```text
x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
```

For time:

```text
t_norm = 2 * (t - t_start) / (t_end - t_start) - 1
```

Physical output scaling should be saved so predictions can be transformed back to physical units.

### Step 5: start with value-only fit

First real-data model should be value-only:

\[
\mathcal{L} = \mathrm{MSE}(\hat{y}, y)
\]

or if uncertainties are clean and trustworthy:

\[
\mathcal{L} = \frac{1}{N}\sum_i \frac{(\hat{y_i}-y_i)^2}{\sigma_i^2}
\]

But start with ordinary MSE if the data loading/scaling is still uncertain.

---

## 13. Heteroscedastic ISR loss plan

Eventually, use reported radar uncertainty.

For one variable:

\[
\mathcal{L}_{data,\sigma}
= \frac{1}{N}\sum_i \frac{(\hat{y_i}-y_i)^2}{\sigma_i^2 + \epsilon}
\]

Potentially include log variance term if treating as Gaussian NLL:

\[
\mathcal{L}_{NLL}
= \frac{1}{2}\sum_i \left[\frac{(\hat{y_i}-y_i)^2}{\sigma_i^2} + \log(\sigma_i^2)\right]
\]

But if sigma comes from the radar fitter and is not learned, the constant/log term may or may not matter depending on what is being optimized. Need to decide carefully.

Use clipping or floors:

```text
sigma_eff = max(sigma, sigma_floor)
```

because tiny uncertainties can dominate training.

---

## 14. Output scaling for ISR

Do not blindly train raw physical values.

Possible choices:

### Electron density

Use:

```text
y = log10(Ne)
```

or normalized log density.

### Temperatures

Possibly use normalized values:

```text
Te_scaled = (Te - mean) / std
Ti_scaled = (Ti - mean) / std
```

or robust scaling using median/IQR.

### Velocity

Could use standardized velocity, but must handle sign.

Important:

```text
If transforming y, transform sigma consistently if using uncertainty-weighted loss.
```

For log transform approximately:

\[
\sigma_{\log_{10} y} \approx \frac{\sigma_y}{y \ln 10}
\]

---

## 15. Coordinate choices for ISR

Initial options:

```text
beam/range/time coordinates
geographic/cartesian coordinates
magnetic coordinates
field-aligned coordinates
```

For first implementation, choose the simplest coordinate system that exists cleanly in the data.

Possible first experiments:

```text
range-time for a single beam: coords = [range, time]
altitude-time for a beam or averaged beam: coords = [altitude, time]
2D spatial slice at one time: coords = [x, z] or [lat, alt]
3D volume at one time: coords = [x, y, z]
```

Do not start with a complex coordinate transform unless needed.

---

## 16. Train/validation split for real data

Need to avoid only evaluating on training points.

Possible splits:

```text
random withheld points
withheld time intervals
withheld beam(s)
withheld range gates
withheld altitude layers
```

For ISR project relevance, the most important are:

```text
withheld-beam validation
withheld-time validation
```

But first debug with random withheld points.

---

## 17. Baselines for real data

Need interpolation baselines.

Possible baselines:

```text
nearest neighbor
linear interpolation
cubic / spline where appropriate
Gaussian smoothing / kernel regression
radial basis function interpolation
ordinary kriging / Gaussian process if feasible
simple temporal interpolation
```

The INR should not be claimed better until compared against simple interpolation under the same withheld tests.

---

## 18. Uncertainty and reliability later

The project eventually wants uncertainty/reliability outputs.

Candidate methods:

```text
MC dropout
deep ensembles
conformal prediction
calibrated residual models
```

Important warning:

```text
Uncertainty does not rescue a bad mean predictor.
```

The mean prediction should be competitive with interpolation baselines first.

Then uncertainty adds value by saying where the prediction is reliable, data-constrained, or prior-dominated.

---

## 19. User preferences / interaction style for next chat

The user wants:

```text
step-by-step implementation
no giant jumps
no assumptions about code state
ask for current files if needed
be explicit about what is known vs assumed
avoid overexplaining when the question is narrow
avoid repeating unasked explanations
```

The user is sensitive to hallucinated code state. If the assistant has not seen the current file, it must say so.

Do not claim “your file has X” unless the current file was pasted/uploaded or shown in the chat.

The user wants practical code, but also wants to understand each step.

Preferred sequence:

```text
1. explain the concept briefly
2. give code
3. explain only the relevant pieces
4. wait for test result
```

---

## 20. Immediate opening prompt for the next chat

The next chat can start with something like:

```text
We are moving from the image INR experiments to real ISR/radar data. We already have a working SIREN-style MLP INR for coordinate-value data, and the best baseline is value-only SIREN with first_omega_0=30 and hidden_omega_0=30. I want to start with a simplified real-data case, probably 2D or 3D, not full 4D yet. Help me inspect the data, build a dataset class that returns coords, values, and uncertainties, and train the first value-only INR before adding heteroscedastic loss or derivative priors.
```

The first thing the next assistant should ask for is:

```text
What is the real data file format and what variables/columns are available?
```

or, if a file is uploaded:

```text
Let's inspect the file structure first before writing model code.
```

---

## 21. Do not forget

Key conclusions to carry forward:

1. Use coordinate-value tuples, not necessarily dense tensors.
2. Time can be an input coordinate; it should not be forced into a tensor depth axis unless using a CNN-style model.
3. Start real data with value-only loss first.
4. Add heteroscedastic weighting only after data scaling and uncertainty interpretation are clear.
5. Add derivative priors later, not at the beginning.
6. Keep Laplacian out of the default ISR design.
7. Compare against interpolation baselines.
8. Save metric histories to CSV.
9. Normalize coordinates and outputs carefully.
10. Be cautious with omega values; SIREN frequency scale is a real hyperparameter.

