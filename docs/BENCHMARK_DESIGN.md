Created: 2026-08-10
Last edited: 2026-08-10
Status: Prepared; smoke only is authorized during implementation

# Benchmark design

## Evidence boundary

Smoke cases test paths and output schemas. Pilot cases estimate plausible timing,
memory, and numerical behavior. Only the later repeated long study can support a
benchmark analysis, and even that will report empirical results rather than claim
implemented plasma physics or unconditional conformal guarantees.

## Controlled synthetic truth

Two inspected formulations remain separately named:

- `gaussian_3d`: a moving three-dimensional Gaussian feature on a constant
  background;
- `chapman_f2_reference`: horizontal Gaussian motion with Chapman-like vertical
  background and feature envelopes.

The canonical integrated target averages **linear** electron density over a centered
window using a declared composite trapezoidal rule, then converts the mean to log10
density. Exact 120- and 300-second synthetic products are controlled experimental
conditions. Instantaneous midpoint truth is retained separately for temporal
smearing; it is not substituted for the integration-product target.

Independent synthetic evaluation uses new seeded physical-coordinate points that
are absent from fitting and collocation. It reports log10- and linear-density RMSE,
MAE, bias, R², relative-error quantiles, and analytic first-derivative errors.

## Real PFISR products

The products remain scientifically labelled nominal 2-minute and nominal 5-minute.
Their ordinary stored durations are 127/128 seconds and 303/304 seconds,
respectively. A single truncated final record occurs in each file (79 and 108
seconds). This is endpoint metadata, not a third product class. Matched studies use
the same physical UTC window and explicitly exclude records below 80% of their
file's median integration duration. The underlying processing reason for the stored
cadences is not inferred from the file alone.

Both products use the same reader, group splitting, trainer, and metric code.
Comparisons record exact boundaries, retained counts, exclusions, observation
density, and actual integration-duration distributions. Training remains unweighted
MSE unless a separately validated uncertainty-weighting comparator is later added;
raw uncertainty is retained and never replaced with invented finite values.

## Questions represented by the long manifest

1. What independent error baseline is obtained from instantaneous, 120-second, and
   300-second synthetic observations without a derivative prior?
2. How do data-only, legacy diagonal, anisotropic-Huber, and full spatial-Hessian
   formulations compare on independent truth and derivative health?
3. How do observed-coordinate, domain-wide Sobol, and support-aware collocation, and
   low versus high collocation counts, affect error, runtime, memory, and collapse?
4. How do random beam, clustered beam, time-block, and joint beam/time withholding
   change empirical coverage and width by beam, altitude, time, and support distance?
5. Across three repeated group splits, does empirical coverage approach the nominal
   level, and where does it fail?
6. Under a shared physical window and metric implementation, how do the nominal
   2-minute and 5-minute PFISR products differ in held-out reconstruction and
   empirical conformal behavior?

Every template is repeated for seeds 0, 1, and 2 through deterministic manifest
expansion. No unrelated Cartesian product of available configuration knobs is
created.

The scientifically matched baseline for the implemented 4D derivative-prior study
is the 4D data-only model under the same observations, split, architecture, seed,
and evaluation code. The copied working 3D code remains an operational compatibility
baseline, but it is not silently treated as a matched dimensional comparator because
its historical experiments use a different data contract. Consequently, this study
can test whether derivative-regularized 4D improves over data-only 4D; it cannot by
itself support a claim that 4D is superior to a separately trained 3D model.

## Interpretation gates

A derivative-prior reduction is not scientific success by itself. Synthetic claims
require independent truth metrics; real-data claims require untouched held-out
groups. Coverage is always empirical and accompanied by its group unit,
exchangeability assumption, and correlated-radar limitation.
