Created: 2026-10-07
Last edited: 2026-10-07
Status: Explicit clean synthetic spatial route; neural verification pending user execution

# Integration-aware xyt reconstruction results

`python -m inr_isr.results_xyt --help` exposes separate `prepare`, `train`,
`verify-smoke`, `analyze` and `final-evaluate` commands. Existing native recipes
retain their implementations. This route requires an explicit case JSON matching
`results_xyt/config.py`; case choices and execution budgets belong to the operator.

Preparation verifies the original radar metadata and saved geometry/time tables,
generates clean analytic labels for training/validation only, fits target limits
on training rows and saves fixed development coordinates. Reserved observations
have identities/coordinates/integration intervals but no density labels. The
metadata input supplies sampling, not measured radar density targets.

The latent network output is normalized instantaneous log10 electron density.
The observation calculation first inverse-scales each node, then computes the
log10 of the weighted **linear** density average over the actual exposure.
Thirty-three normalized trapezoid weights and logsumexp preserve stable gradients.
Training accumulates correctly weighted microbatch gradients of the full
training-observation mean. No derivative/adaptive-prior branch is present.
Full validation beam-macro log10 RMSE selects `best_validation.pt`; exact ties
retain the earlier checkpoint. `last.pt` separately saves the final update.

CUDA is required by this route. Float32, AMP off and all fixed/adaptive prior
settings disabled are schema requirements. `--smoke` changes only the update
cap to two, training-wall cap to 120 seconds, and validation frequency to one.
The wall budget is checked at update boundaries; an in-flight update and final
validation/save can extend it. There is no automatic extension or resume.

`analyze` compares the selected checkpoint with a training-mean log-density
constant and linear time-per-beam followed by barycentric spatial interpolation.
The interpolation uses integration averages as midpoint nodes, an acknowledged
approximation. All observation predictions use matched integration averaging.
Endpoint-constant extensions are confined to observation exposures and reported
separately; instantaneous queries receive no extension. Support masks/counts,
pooled and beam-macro errors in log/physical units, near-path scores, sampled
peak/width diagnostics, four figures and their numerical inputs are saved.
Undefined/truncated widths remain flagged. These are development diagnostics.

Final evaluation is a separate command requiring a supplied freeze JSON with
exact keys `comparison_frozen` (true), `config_identity`, `checkpoint_sha256`,
`operator` and `frozen_utc`. The gate precedes final target/query generation.
Training, smoke and development analysis do not create this record.

Non-neural checks can run explicitly with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_results_xyt_non_neural.py -q
```

Those tests reject neural-module construction/calls. `verify-smoke` is a neural
command for the operator: it checks actual CUDA gradients, unequal microbatch
equivalence, reloaded full-validation scoring and best/last checkpoint selection.
Passing arithmetic tests does not establish GPU execution, convergence or
reconstruction accuracy. No full existing test suite was executed for this work.

Every output must use a fresh directory. A standalone supervising process should
also save UTC start/end, argv/cwd, exit/signal status, input/config/source/runtime
identities and recipes outside bulk outputs, so failures and lost outputs remain
recoverable. This command module accepts `--context` for scientific-process
metadata; the supervisor must also capture failures before Python starts.
