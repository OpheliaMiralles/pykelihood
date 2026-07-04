# PR 5 WIP Assessment

## Recommendation

Do not continue directly on top of the current PR 5 WIP implementation.

Keep the branch as a reference, but start a clean PR 5 branch from `37a34fd` and selectively reuse the good pieces. The current WIP is useful as a prototype and learning artifact, but it is not a good final base.

## Why This Is Not A Full Restart From Zero

The current WIP is pointed in the right broad direction:

- distributions are moving away from the old `Parametrized` execution model
- `pykelihood/likelihood.py` exists
- `pykelihood/parametric/fitting.py` exists
- broad SciPy wrapper generation is being restored
- temporary `GEV` / `GPD` compatibility is being considered
- profiler compatibility accessors were identified as necessary

Those pieces should inform the clean implementation.

## Why The Current WIP Should Not Be Continued As-Is

The current branch has several structural problems that make fix-forward risky:

- tests do not collect because `pykelihood/distributions/base.py` contains orphaned fitting code with an `IndentationError`
- there are two fit-result concepts: `Fit` in `distributions/base.py` and `FitResult` in `parametric/fitting.py`
- `fit_mle(..., **fixed_values)` is missing or only partially implements fixed-parameter semantics
- test edits removed or skipped valuable behavior instead of porting it
- explicit-state semantics are muddy because `with_state` replaces parameter nodes instead of cleanly evaluating the model with state
- expression/effect-valued parameters likely do not evaluate correctly in all paths
- required SciPy shape parameters are not handled according to the migration plan
- compatibility accessors exist, but are not yet principled projections from `ParameterLayout`, `State`, and fit results

The most concerning test losses are around:

- shared parameters in distribution trends
- fixed refits
- profiler behavior
- confidence intervals
- reparametrization behavior
- effect-valued distribution parameters

These are important behaviors, not optional cleanup.

## Suggested Path

1. Keep `distribution-revamp` as an archive/reference branch.
2. Start a clean PR 5 branch from `37a34fd`.
3. Reuse the broad SciPy wrapper list and naming/alias work where it is correct.
4. Reuse the idea of `fit_mle` and `FitResult`, but make `FitResult` the single fit-result type.
5. Implement `Distribution.fit(...)` only as a deprecated forwarder to `fit_mle`.
6. Keep compatibility accessors as deprecated projections from the new graph/state model.
7. Start from old tests and port behavior deliberately; do not delete tests just because their old syntax no longer fits.
8. Add a profiler smoke test before the profiler rewrite, because profiler continuity is non-negotiable.

## First Acceptance Target For The Clean PR 5

Before expanding scope, the clean PR 5 should reach this boring baseline:

- tests collect
- `Normal()` evaluates `pdf`, `logpdf`, `cdf`, `ppf`, and `rvs`
- `fit_mle(Normal(), data)` works
- `Normal().fit(data)` forwards to `fit_mle`
- fixed `loc` fitting works
- `FitResult` exposes the deprecated accessors needed by downstream code and the current profiler
- one fast `pykelihood.profiler.Profiler` smoke test passes through the compatibility bridge
- broad SciPy wrapper coverage is restored only after the core path is stable

## Final Decision

Preserve the current WIP for reference, but restart PR 5 implementation from the last stable effects branch. Cherry-pick ideas, not the whole commit stack.
