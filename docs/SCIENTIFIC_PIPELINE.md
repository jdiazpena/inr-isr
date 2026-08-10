# Scientific and Computational Pipeline

## 1. Reconstruction Objective

The project represents electron density with a coordinate neural network:

```text
f_theta(x_norm, y_norm, t_norm) -> normalized log10(Ne)
```

The primary model is a SIREN-style implicit neural representation. Hidden activations
are sine functions, and linear layers use the existing SIREN initialization. The
default architecture remains 256 hidden features, three hidden layers,
`first_omega_0 = 5`, and `hidden_omega_0 = 5`.

The model reconstructs the plasma field represented by the radar product. For
synthetic integrated measurements, the integration-averaged field is therefore the
ground truth. An unavailable instantaneous midpoint field is a contextual diagnostic,
not a model-selection target.

## 2. Input Coordinates and Target

Both synthetic and real datasets normalize each coordinate independently to `[-1, 1]`:

```text
q_norm = 2 (q - q_min) / (q_max - q_min) - 1
```

Electron density is represented as `log10(Ne)` before target normalization. This
improves numerical scale, but physical reconstruction errors are computed after
converting predictions back to linear `Ne`.

Synthetic observations come from `synthetic_plasma.py`. Real observations come from
`amisr_h5_reader_3d.py`, which selects one range gate per beam inside an altitude
band and produces radar-centered `x`, `y`, and time coordinates.

## 3. SIREN Data Fit

At measured coordinates, the data objective is:

```text
L_data = mean((f_theta(q_i) - y_i)^2)
```

Full-batch training is used when `batch_size` is zero or at least the number of
measurements. Otherwise, each step samples measurements without replacement.

## 4. Collocation Support

Derivative priors are evaluated on a Cartesian x-y grid at every selected observation
time. Grid points farther than a support radius from all measurements are removed.
The radius is the median nearest-neighbor measurement spacing multiplied by
`nearest_radius_factor`.

Collocation points do not add synthetic density targets. They only identify where the
continuous INR derivatives are regularized or diagnosed.

## 5. Soft Curvature Priors

Spatial curvature is the squared Frobenius norm of the x-y Hessian:

```text
L_xy = mean(f_xx^2 + 2 f_xy^2 + f_yy^2)
```

Temporal curvature is:

```text
L_t = mean(f_tt^2)
```

Derivatives are taken with respect to normalized coordinates. These losses prefer a
smooth interpolant while allowing linear temporal evolution. They are not a PINN and
do not enforce continuity, momentum, electrodynamics, or another governing PDE.

The total objective is:

```text
L_total = L_data + lambda_xy L_xy + lambda_t L_t
```

## 6. Adaptive Reference-Ratio Weights

Reference mode maintains exponential moving averages of the data and raw curvature
losses. Its stabilized reference is:

```text
L_ref = max(EMA(L_data), epsilon_data)
```

Target weights are:

```text
lambda_xy_target = target_xy_ratio L_ref / EMA(L_xy)
lambda_t_target  = target_t_ratio  L_ref / EMA(L_t)
```

The targets are clamped to configured limits and blended into the active base lambda.
A warmup and regularization ramp prevent the priors from dominating before the SIREN
has begun fitting measurements.

## 7. Curvature and Gradient Diagnostics

A fixed collocation probe tracks `f_xx`, `f_xy`, `f_yy`, and `f_tt` RMS, maximum,
mean absolute value, exact-zero fraction, and near-zero fraction. These measurements
distinguish a genuinely smooth solution from a coding path that stopped calculating
derivatives.

Optional parameter-gradient diagnostics separately measure the gradient norm produced
by the data, weighted spatial prior, weighted temporal prior, and total objective.
These are diagnostics only and do not modify optimizer gradients.

## 8. Outputs and Checkpoints

Each run records:

- `run_config.json`: effective arguments;
- `history.csv`: losses, lambdas, ratios, derivative health, and measured-point error;
- best-total and best-data checkpoints after the regularization ramp;
- `model_final.pt`: final SIREN parameters, scalers, configuration, and metrics;
- measured-point predictions and optional dense plots/CSVs.

Dense synthetic analysis uses `synthetic_analyze_reconstruction.py`. It reports both
log-space and linear-density errors, compares gradients when enabled, and preserves
midpoint truth only as a secondary integration diagnostic.

## 9. Synthetic and Real Entry Points

The active synthetic and diagnostic real-radar trainers are thin data adapters around
the same `training_engine.py`. Consequently, model construction, loss equations,
adaptive weighting, random ordering, checkpointing, and history fields cannot drift
between the two workflows unnoticed.

The fixed-lambda legacy trainers remain available for reproducing earlier experiments.
They share low-level utilities but retain their historical optimization loops and
output schemas.
