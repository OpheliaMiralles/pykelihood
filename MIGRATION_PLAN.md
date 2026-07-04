# pykelihood Migration Plan

This document describes a practical path from the current `master` branch to the redesigned branch.

The target has moved since this plan was first written. The main architectural center is now:

- graph nodes in `expr.py`
- explicit parameter state in `state.py`
- covariate-dependent effects in `effects.py`
- point-estimate fitting in `parametric/`
- Bayesian fitting in `bayesian.py`
- plain SciPy wrappers plus a separate distribution-level reparametrization layer

The goal is not to preserve every current behavior during migration. The goal is to make each pull request:

- focused on one coherent behavior change
- reviewable without understanding the whole redesign at once
- complete enough that tests in that PR can move with the code change

Each PR may break API where the PR explicitly says so. It should avoid breaking multiple unrelated behaviors at once. The profiler is the main exception: until PR 8 rewrites it, earlier PRs must preserve enough compatibility for the existing profiler to keep working.

This plan should be usable even when the redesign branch is not checked out on disk. A migrator should be able to work mainly from:

- the current `master` code
- the behavior and tests described here
- occasional `git show` or `git diff` peeks at the redesign branch when needed

So the primary contract for each PR is not "edit these exact files". The primary contract is:

- land these behaviors
- replace these tests
- keep these adjacent behaviors working

Module names in the plan are only likely implementation areas.

## Migration Principles

- Decide early whether a legacy feature is still a migration target. If not, remove it in the subsystem PR that makes it obsolete instead of dragging it through later steps.
- Prefer local removal to standalone cleanup. If a legacy feature only becomes clearly unnecessary once a subsystem is rewritten, drop it in that subsystem PR instead of creating a separate pre-cleanup PR.
- Keep old and new internals separate until the public cutover. `master` should not mix two execution paths in the same public methods for long.
- Evolve public concepts where possible. `Parameter`, `Distribution`, kernels/effects, and fitting should remain recognizable even when their implementation changes completely.
- Keep SciPy wrapping and reparametrization separate. Wrapping a SciPy distribution and reparametrizing a `pykelihood` distribution are now two different mechanisms.
- Keep compatible read-only/introspection surfaces when they can be derived cleanly from the new model. Deprecate them when they name old concepts, but do not remove them just because they are no longer core internals.
- Replace tests in the same PR that removes or rewrites the behavior they describe.
- Do not ship migration scaffolding. Design notes and temporary shims are allowed during the transition, but the final package should not keep them unless they still add user value.

## Scope Decisions Before Migration

The redesign no longer aims to preserve all of `master`.

The following should be treated as first-class cleanup candidates rather than porting requirements:

- `pykelihood/visualisation/*`
- the old `ParametrizedFunction` public abstraction
- highly specialized built-in effects that do not justify staying in the reference catalog:
  - `exponential_ratio`

These are either out of scope for the redesign, too specialized to justify keeping, or clearly superseded by the new architecture.

The migration should keep the following capabilities alive throughout the rewrite:

- plain distributions and fitting
- broad SciPy wrapper coverage with native SciPy parameterizations
- the existing distribution vocabulary:
  - `pdf`
  - `logpdf`
  - `ppf`
  - `rvs`
- effect-building primitives plus a small catalog of emblematic effects
- broadly useful regression-style helpers, even if they are temporarily implemented on legacy internals until the effect layer catches up
- profile likelihood tooling
- reparametrized distributions
- metrics

The migration should also preserve downstream-friendly introspection where possible:

- `params_names`
- `flattened_params`
- `optimisation_params`
- `optimisation_param_dict`
- `flattened_param_dict`
- `param_mapping`

These names come from the old architecture, so they should be treated as deprecated compatibility accessors once the new core is public. They should be implemented by projecting from `Node`, `ParameterLayout`, `State`, and fit results, not by keeping `Parametrized` as the execution model.

## Master To Target Module Map

The user-facing implementation on `master` is concentrated in:

- `pykelihood/parameters.py`
- `pykelihood/distributions/base.py`
- `pykelihood/distributions/custom.py`
- `pykelihood/distributions/scipy.py`
- `pykelihood/kernels.py`
- `pykelihood/metrics.py`
- `pykelihood/profiler.py`

The redesigned branch splits those responsibilities across:

- `pykelihood/expr.py`
- `pykelihood/state.py`
- `pykelihood/effects.py`
- `pykelihood/bound.py`
- `pykelihood/likelihood.py`
- `pykelihood/parametric/__init__.py`
- `pykelihood/parametric/profiler.py`
- `pykelihood/reparametrization.py`
- `pykelihood/distributions/base.py`
- `pykelihood/distributions/custom.py`
- `pykelihood/distributions/scipy.py`
- `pykelihood/kernels.py` as a thin effect-oriented compatibility layer
- `pykelihood/metrics.py`
- `pykelihood/bayesian.py`
- `pykelihood/backends/*`

The largest structural change is that `parameters.py` is no longer the engine of the library. Its old responsibilities are split into graph nodes, explicit state, fitting, and distribution wrappers.

## Temporary Compatibility Shims

To reduce cutover pain, two temporary shims are worth allowing:

- `pykelihood/parameters.py` as a thin re-export layer for `Parameter`, `Constant`, `State`, `ParameterLayout`, and transforms once the new core exists
- `pykelihood/kernels.py` as a thin compatibility layer over `effects.py`

These are migration tools, not architectural targets. A final cleanup PR should remove any shim that no longer serves a real compatibility purpose.

`pykelihood/distributions/base.py` is not a shim. It should remain the stable home of `Distribution`.

The allowed compatibility surfaces are:

- `pykelihood/parameters.py` as a legacy-facing public path over the new graph/state layer
- `pykelihood/kernels.py` as a compatibility layer over the new effect machinery
- deprecated distribution and fit-result introspection accessors when they can be derived from the new graph/state representation

Compatibility belongs at those module boundaries or as read-only projections from the new model. It should not reappear as a second execution model.

## Legacy Test Replacement Map

The old test modules map onto the redesign like this:

- `tests/test_parameters.py` -> foundations, state, and parametric-fit tests
- `tests/test_distributions.py` -> distribution tests plus broad wrapper-generation coverage
- `tests/test_kernels.py` -> effect tests and any remaining kernel-shim coverage
- `tests/test_parametrization.py` -> distribution reparametrization tests
- `tests/test_profiler.py` -> `parametric/profiler.py`
- `tests/test_metrics.py` -> metrics tests on the new explicit-state surface

The redesign test suite is now organized by migration topic:

- `tests/test_foundations.py`
- `tests/test_effects.py`
- `tests/test_distributions.py`
- `tests/test_reparametrization.py`
- `tests/test_inference.py`
- `tests/test_profiler.py`
- `tests/test_metrics.py`
- `tests/test_bayesian.py`
- `tests/test_bayesian_pymc_backend.py`

These names should also be the migration target on `master`. There is no need to preserve temporary `v2` filenames.

What matters is that each PR moves the tests for the behavior it changes, and only that behavior.

When retiring an old test file, do not treat that as permission to drop all of its coverage. Review the old file for valuable cases and carry them forward when they still matter on the new architecture. In particular:

- delete tests whose semantics are tied to the old architecture and no longer make sense
- adapt tests whose scenario still matters but whose API or result shape has changed
- port edge cases into the new topic files when they still cover meaningful behavior

This matters more than preserving filenames. The goal is to replace obsolete tests, not to lose subtle coverage by accident.

## How To Execute This Plan

For each PR, the preferred workflow is:

1. Write or move the target tests first.
2. Make the smallest implementation change that makes those tests pass.
3. Keep unrelated legacy paths alive unless that PR explicitly removes them.
4. Only after the behavior works, decide whether the implementation should also be moved into its target module shape.

This is especially important for the early PRs. They should be driven by functionality, not by trying to recreate the target filesystem layout too early.

Two practical rules follow from this:

- Prefer partial class ports over wholesale copies from the redesign branch. If a target class has ten methods but the current PR only needs two of them, only port the two methods and the minimum supporting structure.
- Keep tests importable by topic. Each migration topic should have a focused test entry point so the migrator can work on one behavior at a time without dragging in half-finished adjacent functionality.

That second rule may require a small amount of test reorganization on `master` before or during the migration. That is worthwhile. The migration will be much easier if each topic can be validated independently.

## Completed Early Steps

### Step 1 Completed: Node Core

Landed in `4b78952` (`Parameter expressions`, PR #70).

What actually landed:

- `pykelihood/expr.py` now exists as the graph foundation.
- constants, parameters, and arithmetic composition are available through graph nodes.
- deterministic child traversal and shared-parameter identity are tested in `tests/test_foundations.py`.
- the legacy `parameters.py` surface still exists, so this was not a public cutover.

This means the migration no longer needs a future PR for a first node layer. That work is already part of the base branch.

### Step 2 Completed: Explicit State and Layout

Landed in `c083a6a` (`Add explicit parameter state and layout helpers`, PR #72).

What actually landed:

- `pykelihood/state.py` now provides explicit `State`, `ParameterLayout`, and transform-aware flatten/unflatten helpers.
- state/layout behavior is already covered by dedicated tests.
- the old fitting path was left in place, so the new state layer can coexist with legacy APIs for now.

This means later migration work can treat explicit state and parameter layout as existing infrastructure, not as a planned future step.

### Steps 3 and 4 Completed Together: Effects, Helpers, and Kernel Compatibility

Landed together in `37a34fd` (`Introduce effects as generalized kernels`).

What actually landed:

- `pykelihood/effects.py` now exists and is the real implementation surface for effects.
- `pykelihood.kernels` still exists, but now serves as a compatibility-oriented layer over effects instead of being the primary implementation.
- a new `tests/test_effects.py` covers the direct effect API.
- `tests/test_kernels.py` still exists as compatibility coverage.
- higher-level helpers such as polynomial, regression, and categorical-style kernels were carried through in the same compatibility-oriented PR rather than in a later clean break.

What this means for the remaining migration:

- there is no longer a meaningful split between a future “effects core” PR and a later “regression/categorical helpers” PR.
- cleanup of the old kernel surface should now happen when the broader public cutover makes it safe, not in a dedicated earlier PR.
- later PRs should assume the new effect machinery exists, but also remember that compatibility debt is still present in `pykelihood.kernels` and `pykelihood.parameters`.

## Architectural Boundaries For The Remaining Steps

The codebase is intentionally mixed:

- `expr.py`, `state.py`, and `effects.py` are real new-core modules and already have topic tests.
- `distributions` and point-estimate fitting are in the middle of the PR 5 rewrite. Some new-core files exist, but the branch is not complete until tests collect and the duplicate/legacy fitting paths are resolved.
- profiler and metrics still mostly describe the old fitted-distribution conventions.
- `tests/test_distributions.py`, `tests/test_parametrization.py`, `tests/test_inference.py`, `tests/test_profiler.py`, and `tests/test_metrics.py` may contain a mix of new-core expectations and old compatibility assertions. Treat old `Parametrized`-specific assertions as migration debt unless a PR explicitly keeps them as compatibility coverage.

The remaining migration work should follow these guardrails:

- Do not make new `Distribution` code inherit from `Parametrized`. Cut that chain cleanly.
- There is one distribution execution model. The new distribution classes replace the old ones; they are not a parallel tree to be cut over later.
- Keep compatibility at module boundaries when it is cheap, not inside the execution model.
- Do not solve future extensions while porting the core. In particular, keep the following out of scope for the remaining migration steps unless a PR explicitly says otherwise:
  - partial binding / `partial_bind`
  - transformed distributions such as `exp(Normal(...))`
  - generic algebra over distributions
  - autodiff / numerical backends
  - diagnostics beyond the profiler and current metrics
  - richer Bayesian predictive tooling

An implementer should read the remaining PRs below as a concrete path from this mixed state, not as a request to recreate the full redesign branch in one jump.

## Profiler Preservation Contract

The profiler is not optional migration debris. It is one of the most valuable internal capabilities in the library and must keep working until the dedicated profiler rewrite lands.

Before PR 8, do not "simplify" the profiler by deleting old surfaces it depends on. Keep enough deprecated compatibility for `pykelihood.profiler.Profiler` to run on the new distribution/fitting core.

The pre-PR-8 compatibility bridge must support:

- `Distribution.fit(data, score=..., x0=..., scipy_args=..., **fixed_values)` as a deprecated forwarder to `fit_mle`
- fitted result `.fit(...)` as a deprecated forwarder that refits the fitted model with optional fixed values
- score functions that accept the old profiler call shape `score_function(fitted_or_distribution, data)`
- `flattened_params` on distributions and fit results
- `optimisation_params` on distributions and fit results
- `optimisation_param_dict` on fit results
- `flattened_param_dict` on fit results, with values exposing `.value`
- `param_mapping()` on fit results
- attribute forwarding from fit results to fitted model parameters, such as `fit.loc`

These are compatibility requirements, not new-core design requirements. Implement them as projections from `FitResult.model`, `FitResult.state`, `ParameterLayout`, and the model graph. Do not resurrect `Parametrized` to satisfy them.

Minimum profiler smoke coverage must remain active before PR 8:

- construct a fitted model with `Distribution.fit(...)`
- instantiate `pykelihood.profiler.Profiler(fit, data)`
- read `profiler.optimum`
- run `profiler.profiles` for a single parameter on a small grid or equivalent fast path
- compute `confidence_interval(...)` for one scalar parameter when the numeric problem is stable enough for CI
- verify fixed-parameter profiling/refitting still fixes the requested parameter

Use small deterministic datasets and a stable distribution for smoke tests. If old `GEV`-specific tests become unstable or depend on a legacy custom class, keep their behavioral intent but move the profiler smoke to a plain SciPy wrapper or a local reparametrized wrapper. Do not drop profiler coverage because `GEV` moved.

## Remaining PR Sequence

The remaining work should be read as a close-out plan from the current mixed branch, not as a restart from `master`.

Important current-state assumptions:

- Steps 1-4 are complete.
- PR 5 has been partially implemented on the current branch: the new distribution base, broad SciPy wrapper list, `likelihood.py`, and `parametric/fitting.py` exist.
- PR 5 is not done until its implementation is coherent, tests collect, and the fitting contract is settled.
- Backwards compatibility was kept more than the original plan expected. That is acceptable, but it changes the remaining goal: compatibility now needs to be made explicit and thin, not silently removed.

### PR 5 Close-Out: Distribution And Parametric Fit Core

Goal:

- Finish the replacement of the old `Parametrized`-based distribution and point-estimate fitting engine with the node/state-based core.

This PR is about making the new core correct. It is not the public API cleanup PR.

Required behavior:

- `pykelihood.distributions.base.Distribution` is the real base class for new distributions and does not inherit from `Parametrized`.
- `ScipyDistribution` and generated SciPy wrappers run on the new distribution base.
- Distribution methods accept explicit state:
  - `pdf(x, *, state=None)`
  - `logpdf(x, *, state=None)`
  - `cdf(x, *, state=None)`
  - `ppf(q, *, state=None)`
  - `rvs(size=None, *, state=None, random_state=None)`
- `sf`, `isf`, `logcdf` or `log_cdf`, and `logsf` or `log_sf` remain available where the underlying SciPy distribution supports them.
- Constructors accept `Node | ArrayLike | None`.
- Explicit literals become constant nodes.
- Omitted `loc` and `scale` become free parameters with defaults `0.0` and `1.0`.
- Required SciPy shape parameters must be provided explicitly. Do not invent defaults for `beta`, `gamma`, `pareto`, `genextreme`, `genpareto`, etc.
- Effect-valued and expression-valued parameters evaluate through the normal child-node/state machinery.
- `Normal` remains available as a public alias over the plain SciPy `norm` wrapper.
- `Bernoulli` and `TruncatedDistribution` are the only native custom distributions that should be needed before later PRs.
- Broad continuous SciPy coverage exists in a boring, systematic wrapper list.
- Wrapper naming uses one helper plus a small explicit alias table.
- Plain SciPy wrappers use native SciPy parameter names and meanings. They should not permanently absorb statistical reparametrization logic.
- `log_likelihood`, `negative_log_likelihood`, and `fit_mle` work on the new distribution/state surface.
- `fit_mle` returns `FitResult`; it does not mutate the model.
- Fixed-parameter refits work through explicit state/layout handling.
- `Distribution.fit(...)` may exist in this PR only as a thin compatibility forwarder to `fit_mle`.
- Deprecated distribution introspection accessors remain where they can be computed from the new graph:
  - `params_names` returns the public distribution parameter names
  - `flattened_params` returns the parameter/expression nodes in deterministic parameter-name order
  - `optimisation_params` returns free `Parameter` nodes discovered through the distribution graph, deduplicated in `ParameterLayout` order
- These accessors must not call old `Parametrized` traversal or require mutation-style fitting.

Current branch cleanup checklist:

- Fix any syntax or indentation errors in `pykelihood/distributions/base.py`; no orphaned fitting code should remain after `with_state`.
- If a compatibility `.fit(...)` method is kept, implement it explicitly as a small forwarder to `pykelihood.parametric.fit_mle`.
- Remove duplicate fitting implementations from the distribution base. There should be one MLE implementation: `pykelihood/parametric/fitting.py`.
- Make `FitResult` the only new fit-result type. Do not keep a second `Fit` dataclass in `distributions/base.py` unless it is deliberately retained as a legacy alias.
- Make fixed values in `fit_mle(..., **fixed_values)` actually constrain the optimization rather than being ignored.
- Ensure `ParameterLayout` is built from the model graph that remains after fixed values are applied.
- Keep compatibility helpers such as `flattened_params`, `optimisation_params`, and `params_names` as deprecated projections from the new graph. They must not drive the new execution model.
- Add deprecation warnings for old-concept accessors where doing so will not make common downstream reads unusably noisy. Prefer one warning per property access, using `DeprecationWarning`, and cover the warning behavior in focused compatibility tests.
- If a legacy accessor cannot be implemented honestly from the new model, document the gap in this plan before removing it.
- Keep the old profiler running by preserving the compatibility bridge listed in "Profiler Preservation Contract". In particular, `FitResult` must support the read/refit methods that `pykelihood.profiler.Profiler` calls today.

Touch:

- `pykelihood/distributions/base.py`
- `pykelihood/distributions/custom.py`
- `pykelihood/distributions/scipy.py`
- `pykelihood/distributions/__init__.py`
- `pykelihood/likelihood.py`
- `pykelihood/parametric/__init__.py`
- `pykelihood/parametric/fitting.py`
- `tests/test_distributions.py`
- `tests/test_inference.py`

Do not touch:

- public package cutover in `pykelihood/__init__.py`
- `pykelihood/parameters.py` cleanup
- profiler
- metrics
- Bayesian code
- optional backend code

Test requirements:

- Keep valid distribution scenarios in `tests/test_distributions.py`; do not delete coverage just because old syntax changes.
- Put fit-specific assertions in `tests/test_inference.py` when that makes ownership clearer.
- Add or preserve tests for broad SciPy wrapper coverage, wrapper naming/aliases, default `loc`/`scale`, required shape arguments, `rvs(..., random_state=...)`, explicit `state` evaluation, `log_likelihood`, `negative_log_likelihood`, `fit_mle`, fixed-parameter fitting, and one composed/effect-valued model.
- Add compatibility tests for deprecated `params_names`, `flattened_params`, and `optimisation_params` on at least one plain wrapper and one expression/effect-valued model.
- Add a fast profiler smoke test that uses the deprecated compatibility bridge but does not rewrite profiler internals.
- Do not move profiler confidence-interval implementation here.

Accept when:

- tests collect cleanly
- distribution code no longer uses `Parametrized` as its implementation base
- `fit_mle` has one score contract and one implementation
- point-estimate results are separate objects with `model`, `state`, and optimizer metadata
- broad SciPy wrapper coverage works on the new core
- downstream parameter-introspection accessors still return compatible information with deprecation warnings
- existing `pykelihood.profiler.Profiler` can still compute `optimum` and at least one single-parameter profile through compatibility accessors

### PR 6: Public API Cutover And Legacy Test Retirement

Goal:

- Make the graph/state/effect/distribution/parametric stack the default public API while keeping intentional compatibility modules thin.

This PR is about default imports and tests. It is not a core distribution rewrite; PR 5 should already have done that.

Required behavior:

- `import pykelihood` exposes the new normal-use modules:
  - `distributions`
  - `effects`
  - `kernels`
  - `likelihood`
  - `metrics`
  - `parametric`
  - `profiler`
- `pykelihood.__all__` should not re-export old `Parametrized`, `ParametrizedFunction`, `ConstantParameter`, or other legacy internals.
- `pykelihood/parameters.py` becomes a compatibility module over the new core:
  - public: `Parameter`, `Constant`, `State`, `ParameterLayout`, `Transform`, `PositiveTransform`, `ProbabilityTransform`
  - legacy internals may remain importable from the module for now, but they should not be in `__all__`
- `pykelihood/kernels.py` remains as a compatibility layer over `effects.py`, because backwards compatibility was intentionally preserved.
- `Distribution.fit(...)`, if present, is only a forwarder to `fit_mle`.
- Deprecated fit-result compatibility accessors remain when they can be projected from `FitResult.model` and `FitResult.state`:
  - `optimisation_param_dict`
  - `flattened_param_dict`
  - `param_mapping`
  - attribute forwarding to the fitted model
- Deprecated fit-result `.fit(...)` remains as a forwarder to `fit_mle` because the current profiler uses it for fixed-parameter profile refits.
- These accessors should expose compatible values for downstream code, but new tests and docs should prefer `FitResult.state`, `ParameterLayout`, and explicit model attributes.

Concrete current-branch tasks:

- Remove the old `tests/test_parameters.py`; its valuable coverage belongs in `tests/test_foundations.py` and `tests/test_inference.py`.
- Rewrite `tests/test_kernels.py` as compatibility-surface tests:
  - imports from `pykelihood.kernels` still work
  - representative helpers return callable `Kernel` objects
  - `Kernel.__call__()` works
  - `with_covariate(...)` works
  - deprecated read-only introspection such as parameter names or optimization parameters works if it can be derived from the effect graph
  - do not assert mutation through `with_params` or exact old flattened name paths unless that exact spelling is intentionally kept as compatibility
- Update remaining tests to prefer `fit_mle(model, data, ...)` over mutation-style `model.fit(data, ...)`.
- If a test still uses `.fit(...)`, it should be a compatibility-forwarder test, not the main fitting-path test.
- Replace `ConstantParameter` usage in new-path tests with literals, `Constant`, or `Parameter`, depending on what the test is actually proving.
- Add fit-result compatibility tests for `optimisation_param_dict`, `flattened_param_dict`, and `param_mapping`. Mark them as deprecated compatibility behavior, not as the recommended new API.
- Keep or add profiler compatibility tests during the cutover. The public API cutover is not allowed to break `pykelihood.profiler.Profiler`.

Touch:

- `pykelihood/__init__.py`
- `pykelihood/parameters.py`
- `pykelihood/kernels.py` only if the compatibility layer needs export adjustments
- `tests/test_parameters.py`
- `tests/test_kernels.py`
- old-path assertions in distribution/inference tests

Do not touch:

- reparametrization implementation
- profiler behavior
- metrics behavior
- Bayesian behavior

Accept when:

- `import pykelihood` points users at the new core
- old mutation-based fitting is no longer the tested default
- `tests/test_parameters.py` is gone
- `tests/test_kernels.py` tests compatibility behavior only
- compatibility modules are thin boundary layers, not places where new execution logic lives
- fit-result and kernel compatibility accessors are retained as deprecated projections where practical
- `pykelihood.profiler.Profiler` still works through the compatibility bridge after the public API cutover

### PR 7: Distribution Reparametrization Layer

Goal:

- Restore statistical parameterization conveniences as a wrapper over plain `pykelihood` distributions, not inside SciPy wrappers.

Why this exists:

- Many legacy custom names were really reparametrizations.
- PR 5 deliberately keeps plain SciPy wrappers plain.
- Backwards compatibility can preserve useful names, but the permanent mechanism should be a separate reparametrization layer.

Required behavior:

- `pykelihood/reparametrization.py` exists.
- A plain distribution can be wrapped to expose a different public parameter surface.
- Value-level mappings are supported, for example `log_scale -> exp(log_scale)` or `sigma -> scale`.
- Fitting a reparametrized distribution works through `fit_mle`.
- Fixed public parameters work during fitting.
- The wrapper resolves mapped numeric parameter values before calling the base distribution.
- Structural graph rewrites, algebra over distributions, and transformed distributions are out of scope.

Touch:

- `pykelihood/reparametrization.py`
- `pykelihood/distributions/__init__.py` only for deliberate public reparametrized names
- `tests/test_parametrization.py` or renamed `tests/test_reparametrization.py`

Do not touch:

- plain SciPy wrapper semantics
- profiler
- metrics
- Bayesian code

Test requirements:

- renamed public parameters
- fixed values
- value-level derived mappings
- fitting through a reparametrized wrapper
- at least one test-local wrapper, such as a user-facing `LogNormal` over SciPy `Lognorm`, to prove the mechanism without bloating built-ins

Accept when:

- reparametrization is independent of old parameter containers
- SciPy wrappers remain plain
- any temporary wrapper-level reparametrization from PR 5 has either moved here or been deleted

### PR 8: Parametric Profiler

Goal:

- Rebuild profile likelihood tooling on explicit states and `fit_mle`.

This is a high-value subsystem PR. Do not use it as a cleanup bucket. The goal is to make the profiler better while preserving the behavior users already rely on.

Required behavior:

- New implementation lives in `pykelihood/parametric/profiler.py`.
- Legacy top-level `pykelihood.profiler.Profiler` remains importable and delegates to the new implementation.
- The new profiler accepts explicit model/state inputs:
  - preferred construction: `Profiler(model, data, state=result.state)`
  - accepted construction: `Profiler(model, data)` computes its own optimum with `fit_mle`
  - compatibility construction: `Profiler(fit_result, data)` extracts `model` and `state` from the result
- Profile likelihood sweeps use explicit state updates plus fixed-parameter refits through `fit_mle`.
- The profiler has one score contract:
  - default score is negative log likelihood
  - custom score functions receive `(model, data, *, state=None)` if practical
  - compatibility wrappers may adapt old `score_function(fitted_or_distribution, data)` callables, but the core implementation must not probe several signatures at runtime
- Confidence intervals are methods on profiler/result objects, not on mutable fitted distributions.
- Single-parameter profiling remains supported.
- Profiling all free parameters remains supported.
- Fixed-parameter initial fits remain supported and profiled output must not treat fixed parameters as free profiling targets.
- Effect-valued/expression-valued distribution parameters should work if PR 5 fitting supports them; otherwise add a skipped test with a precise blocker.
- Profile output remains tabular and inspectable. A `pandas.DataFrame` result is acceptable and preferred for compatibility.
- Profile output must include:
  - one column per distribution parameter shown in the fitted result
  - a `score` column
  - rows only for finite scores
- Confidence interval search should use the likelihood-ratio threshold from the old profiler unless the PR explicitly changes the statistical contract.

Worker execution order:

1. Add fast tests that lock in the current profiler behavior through the compatibility bridge.
2. Introduce `pykelihood/parametric/profiler.py` with the new explicit-state implementation.
3. Make the top-level `pykelihood/profiler.py` delegate to the new implementation without changing its import path.
4. Port `optimum` first. It must return a fitted result/state and the corresponding score.
5. Port single-parameter `profiles` next. Use a small deterministic grid in tests.
6. Port all-parameter profiling after single-parameter profiling works.
7. Port fixed-parameter profile refits. Verify fixed parameters stay fixed and are not profiled.
8. Port `confidence_interval(...)`.
9. Keep `confidence_interval_bs` as a deprecated alias if it existed before.
10. Remove only profiler compatibility code that is now replaced by a tested delegation path.

Do not make these decisions in this PR:

- Do not redesign the statistical meaning of profile likelihood intervals.
- Do not replace the profiler with a Bayesian or posterior diagnostic.
- Do not change distribution wrapper naming just to satisfy profiler tests.
- Do not restore old custom distributions purely for profiler tests; use plain wrappers or test-local reparametrized models.
- Do not delete `pykelihood.profiler.Profiler`.

Touch:

- `pykelihood/parametric/profiler.py`
- `pykelihood/profiler.py` as a compatibility shim/delegator
- `tests/test_profiler.py`

Do not touch:

- metrics
- Bayesian code
- distribution wrapper coverage

Test requirements:

- `Profiler(model, data)` computes an optimum using `fit_mle`.
- `Profiler(model, data, state=result.state)` uses the supplied reference state.
- `Profiler(fit_result, data)` works for backwards compatibility.
- `profiler.optimum` returns a result/state whose score equals the model log likelihood or configured score at that state.
- single-parameter `profiles` returns exactly one keyed profile when `single_profiling_param` is set.
- all-parameter `profiles` profiles only free parameters, not fixed parameters.
- each profile has a `score` column and one column per fitted distribution parameter.
- profile scores are finite and do not exceed the optimum score beyond numerical tolerance.
- fixed-parameter refits keep the fixed parameter at the requested value.
- `confidence_interval(param)` brackets the optimum for a stable scalar parameter.
- `confidence_interval_bs` remains an alias if retained.
- top-level import `from pykelihood.profiler import Profiler` still works.
- old `GEV`-style tests should use plain SciPy wrappers or a test-local reparametrized model, not resurrect legacy custom continuous classes.
- compatibility tests should assert behavior, not old internal implementation details.

Accept when:

- profiler code no longer depends on fitted distributions mutating themselves
- profiler still supports the compatibility construction and import paths used before PR 8
- confidence intervals come from profiler objects/results
- fixed-parameter profiling works through the new fit core
- `tests/test_profiler.py` describes both the new explicit-state API and the retained compatibility API

### PR 9: Metrics Port

Goal:

- Move metrics onto `FitResult`, explicit states, and ordinary distribution objects.

Required behavior:

- `AIC` and `BIC` operate on `FitResult` or on an explicitly supplied model/state/data triple.
- Predictive scores such as CRPS, Brier score, quantile score, QQ distance, and PP distance operate on a distribution that is already evaluable with its state.
- Metrics do not inspect old parameter containers, mutation history, or legacy fitted-distribution conventions.
- Likelihood helper behavior from PR 5 should be reused rather than reimplemented.
- If an old metrics function accepted a fitted distribution, keep that call shape when it can unambiguously read `FitResult.model` and `FitResult.state`; emit a deprecation warning and delegate to the new implementation.

Touch:

- `pykelihood/metrics.py`
- `tests/test_metrics.py`

Do not touch:

- Bayesian code
- profiler implementation except for using its public result objects if needed

Test requirements:

- likelihood-based metrics on `FitResult`
- predictive/scoring-rule metrics on a distribution plus explicit state or a fitted/bound result
- legacy `GEV` / `GPD` examples should use plain wrappers or local reparametrized models
- deprecated old call shapes that are intentionally retained

Accept when:

- metrics have no dependency on old parameter plumbing
- metrics tests can be understood without knowing `Parametrized`

### PR 10: Bayesian Module

Goal:

- Add Bayesian fitting as a bridge over the stable graph/state/distribution model.

Required behavior:

- MAP and MCMC are separate fitting modes.
- Bayesian result types are separate from `FitResult`.
- `PosteriorResult` exposes posterior samples/state helpers and does not pretend the posterior is a `Distribution`.
- The parametric core does not import Bayesian internals.
- Translation coverage can be narrow. Support only the subset exercised by tests in this PR.

Touch:

- `pykelihood/bayesian.py`
- `pykelihood/__init__.py` only for intentional public exposure
- `tests/test_bayesian.py`

Do not touch:

- optional PyMC backend dispatch
- profiler
- metrics

Test requirements:

- MAP fitting
- MCMC result shape/trace handling if implemented in this slice
- posterior-state helpers
- predictive sampling for the supported subset

Accept when:

- Bayesian fitting exists without changing parametric semantics
- posterior results remain distinct from ordinary fitted-state results

### PR 11: Optional PyMC Backend Integration

Goal:

- Add optional PyMC backend integration after the Bayesian API is stable.

Required behavior:

- `import pykelihood` works without PyMC installed.
- Backend imports are lazy.
- Backend dispatch is declarative.
- The first backend target is PyMC only.
- This is an optional fitting backend, not a numerical backend abstraction.
- No deterministic modeling, profiler, or metrics behavior changes in this PR.

Touch:

- `pykelihood/backends/__init__.py`
- `pykelihood/backends/pymc.py`
- packaging metadata for optional dependencies
- `tests/test_bayesian_pymc_backend.py`

Do not touch:

- deterministic modeling semantics
- profiler
- metrics

Test requirements:

- lazy import without PyMC
- backend dispatch when PyMC is available or mocked
- declarative distribution mapping

Accept when:

- PyMC remains optional
- backend mapping is not hand-written at every call site
- Bayesian public API can dispatch to the backend without making it mandatory

### PR 12: Compatibility Cleanup

Goal:

- Remove migration-only scaffolding while keeping deprecated compatibility surfaces that still have real downstream value.

Required behavior:

- Review every compatibility item intentionally preserved during PRs 5-11.
- Keep `pykelihood.parameters` only if it is still a useful legacy import path; if kept, it must remain a thin shim.
- Keep `pykelihood.kernels` only if it is still useful as a compatibility API over `effects.py`; if kept, it must remain a thin shim.
- Keep deprecated distribution, fit-result, kernel/effect, profiler, and metrics accessors when they are cheap projections from the new model and are likely downstream integration points.
- Remove stale aliases, duplicate fit result classes, migration-only helpers, and tests that exist only to protect transitional internals.
- For every deprecated compatibility surface that remains, make sure:
  - its implementation delegates to the new core
  - its warning text names the preferred replacement
  - tests cover behavior without requiring old internals
- The package layout should read like the redesign rather than a layered transition.

Touch:

- compatibility shims
- temporary re-exports
- docs or tests that mention migration-only paths

Do not touch:

- core behavior
- new feature semantics

Accept when:

- no migration-only shim remains unless it has explicit compatibility value
- no old execution model is reachable from normal public APIs, even through deprecated compatibility accessors
- tests protect supported behavior rather than implementation leftovers

## Not Part Of This Migration

The following are real future topics, but they should not widen the migration PRs:

- partial binding as a first-class feature distinct from full binding
- additional regression conveniences beyond the current effect helpers
- generic transformed distributions beyond the current wrapper taxonomy
- posterior/predictive object refactors beyond the current Bayesian result layer
- autodiff / JAX numerical-backend work
- non-parametric model support

Those should be handled after the redesigned core lands.
