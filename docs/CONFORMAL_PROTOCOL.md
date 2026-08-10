Created: 2026-08-10
Last edited: 2026-08-10
Status: Active conformal protocol

# Split conformal protocol

## Separation from training

Conformal prediction is an evaluation layer, not a second trainer. It consumes a
trained model checkpoint, the checkpoint's exact preprocessing state, and the saved
group split. Calibration observations cannot update weights, select hyperparameters,
or perform early stopping. The checkpoint and supplied split/preprocessing must
match exactly.

## Score and finite-sample quantile

The current score is the absolute residual in inverse-transformed log10 electron
density. For `n` finite calibration scores and miscoverage `alpha`, the selected
one-indexed rank is

`min(n, max(1, ceil((n + 1) * (1 - alpha))))`.

The quantile is the score at that upper order statistic, exactly equivalent to a
valid `higher` selection. Empty sets, non-finite values, mismatched shapes, invalid
alpha, and invalid transformations are errors; none can silently produce a
zero-width interval.

Intervals are symmetric: prediction minus/plus the calibrated quantile. The saved
calibration artifact contains all scores, alpha, selected rank, quantile,
calibration groups/unit, score convention, transform description, and SHA-256 model
checkpoint identity.

## Evaluation and language boundary

The untouched test stage reports empirical marginal point-level coverage and width,
with stratification by beam, time block, altitude bin, and distance from training
observational support. A group bootstrap summarizes sampling variability when at
least two test groups exist. Machine-readable predictions include coordinates,
targets, residuals, interval bounds, identifiers, and support distance.

These are empirical results under a declared calibration/test group unit. Correlated
radar observations may violate point-level exchangeability. Generated reports state
this limitation and never claim unconditional or “guaranteed 95% coverage.”
