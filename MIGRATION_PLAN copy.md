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

Each PR may break API. It should avoid breaking multiple unrelated behaviors at once.

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

- Narrow the scope before rewriting internals. If a legacy feature is not part of the target architecture, remove it early instead of dragging it through every core PR.
- Prefer local removal to standalone cleanup. If a legacy feature only becomes clearly unnecessary once a subsystem is rewritten, drop it in that subsystem PR instead of creating a separate pre-cleanup PR.
- Keep old and new internals separate until the public cutover. `master` should not mix two execution paths in the same public methods for long.
- Evolve public concepts where possible. `Parameter`, `Distribution`, kernels/effects, and fitting should remain recognizable even when their implementation changes completely.
- Keep SciPy wrapping and reparametrization separate. Wrapping a SciPy distribution and reparametrizing a `pykelihood` distribution are now two different mechanisms.
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
- the new distribution vocabulary:
  - `prob`
  - `log_prob`
  - `quantile`
  - `sample`
- effect-building primitives plus a small catalog of emblematic effects
- broadly useful regression-style helpers, even if they are temporarily implemented on legacy internals until the effect layer catches up
- profile likelihood tooling
- reparametrized distributions
- metrics

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

`pykelihood/distributions/base.py` is not a shim. It exists on both branches and should remain the stable home of `Distribution`.

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

## Proposed PR Sequence

### PR 1: Node Core

Goal:

- Introduce the graph foundation without cutting over public behavior yet.

Functional target:

- there is a node concept for constants and parameters.
- parameters have deterministic child traversal and stable identity.
- arithmetic composition exists independently of distributions and fitting.
- nothing in this PR requires users to switch to explicit `State` yet.

Touch:

- `pykelihood/expr.py`
- `pykelihood/parameters.py` only if needed as a temporary public home of the refactored concepts
- foundation tests

Do not touch:

- explicit state
- effects
- distributions
- fitting
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- `Parameter` and constants become graph nodes with deterministic child traversal

Test movement:

- add tests for traversal order, literal normalization, expression arithmetic, and shared parameter identity
- keep old parameter tests alive for the still-active public path

Accept when:

- graph traversal is deterministic
- literal values normalize into constant nodes
- arithmetic composition exists independently of fitting and distributions

### PR 2: Explicit State, ParameterLayout, and Transforms

Goal:

- Add explicit parameter state and flattening on top of the new graph.

Functional target:

- a model graph can expose its free parameters in deterministic order.
- free parameter values can be packed into and unpacked from a flat optimization vector.
- parameter transforms are applied in one place during packing/unpacking.
- the legacy fitting path may still exist, but the new state machinery is usable and tested on its own.

Touch:

- `pykelihood/state.py`
- `pykelihood/parameters.py` only for temporary re-exports if needed
- state-focused tests

Do not touch:

- effects
- distributions
- fitting
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- parameter discovery, flattening, and transforms are centralized in `State` and `ParameterLayout`

Test movement:

- add tests for parameter discovery, deterministic path order, flatten/unflatten, transform application, and initial-state precedence
- keep old parameter tests alive until the public cutover

Accept when:

- `ParameterLayout.from_expr(...)` is deterministic
- flatten/unflatten round-trips through `State`
- transforms live in the layout/state layer rather than the old parameter tree

### PR 3: Effects Core and Kernel Shim

Goal:

- Replace the old kernel machinery with the new effect layer while preserving a compatibility import path.

Functional target:

- simple covariate-dependent modeling works through an effect abstraction.
- `pykelihood.kernels` still exists, but it forwards to the new machinery instead of staying the primary implementation.
- users can define new effects with the new primitives.
- the built-in catalog is intentionally small and limited to reference/common cases.

Touch:

- `pykelihood/effects.py`
- `pykelihood/kernels.py`
- effect and kernel tests

Do not touch:

- distributions
- fitting
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- covariate modeling moves to effect-building primitives plus a small built-in catalog
- the package no longer aims to preserve the old kernel catalog, only a few emblematic effects
- `pykelihood.kernels` becomes a thin alias layer, not the core implementation
- specialized legacy helpers that do not justify a place in the new effect catalog can be dropped in this PR rather than ported

Test movement:

- replace simple kernel tests with effect-oriented tests covering the remaining built-ins:
  - constant
  - linear
  - polynomial
  - exponential
  - gaussian
  - shared parameters
  - bound covariates
- remove tests for specialized helpers that are intentionally not being carried into the new effect layer

Accept when:

- users can build new effects from the new primitives without depending on a large built-in catalog
- the remaining built-in effects serve as reference implementations and common conveniences
- `pykelihood.kernels` is only a compatibility facade
- effects can compose with ordinary graph nodes where expected

### PR 4: Regression and Categorical Helpers On Top Of Effects

Goal:

- Reintroduce or adapt the broadly useful higher-level helpers that should survive the rewrite.

Functional target:

- there is a supported way to express matrix linear predictors.
- there is a supported way to express polynomial regression terms.
- there is a supported way to express categorical effects.
- these helpers are conveniences layered on top of the effect primitives, not separate machinery.

Touch:

- `pykelihood/effects.py`
- `pykelihood/kernels.py`
- helper-focused effect tests

Do not touch:

- distributions
- fitting internals
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- regression-style helpers are rebuilt on top of the new effect layer rather than the old parametrized-function machinery
- helpers that remain should be those with clear modeling value:
  - linear regression / matrix linear predictor
  - polynomial regression
  - categorical effects

Test movement:

- replace the old regression and categorical kernel tests with effect-oriented equivalents
- validate that these helpers compose with the lower-level effect primitives rather than bypassing them

Accept when:

- broadly useful regression and categorical helpers work on the new effect layer
- they are clearly implemented as conveniences on top of the new primitives
- no legacy parametrized-function machinery is needed to support them
### PR 5: Distribution Base and Broad SciPy Wrapper Surface

Goal:

- Rebuild the distribution layer on the new graph/state/effect foundation.

Functional target:

- the primary distribution API uses:
  - `prob`
  - `log_prob`
  - `quantile`
  - `sample`
- old SciPy-style names such as `pdf`, `logpdf`, and `ppf` are not the migration target.
- only genuinely distinct native distributions remain custom.
- standard continuous families come from plain SciPy wrappers rather than hand-written custom classes.
- custom distributions use the new node/state contracts.
- SciPy-backed distributions use the same contracts.
- broad SciPy coverage is restored through wrapper generation rather than hand-written wrappers.
- sampling uses keyword-only `rng`.
- plain wrappers use native SciPy parameter names and meanings.

Touch:

- `pykelihood/distributions/base.py`
- `pykelihood/distributions/custom.py`
- `pykelihood/distributions/scipy.py`
- `pykelihood/distributions/__init__.py`
- distribution tests

Do not touch:

- fitting internals
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- distributions become explicit-state graph nodes
- SciPy wrappers are plain wrappers with native SciPy parameter surfaces
- `Normal` lives with the SciPy wrappers, not as a special native distribution
- most old custom continuous wrappers are removed instead of being ported
- the public distribution vocabulary shifts from `pdf`/`logpdf`/`ppf` to `prob`/`log_prob`/`quantile`

Test movement:

- replace the surviving portions of `tests/test_distributions.py`
- add tests for:
  - the reduced set of native custom distributions
  - wrapper generation over the supported SciPy continuous catalog
  - keyword-only `rng`
  - simple effect-valued parameters
  - the preferred `prob` / `log_prob` / `quantile` spellings

Accept when:

- custom and SciPy-backed distributions work on the new core
- sampling and density evaluation use explicit state
- the broad one-line SciPy wrapper surface is restored
- the package no longer depends on the old `Parametrized` hierarchy for distributions

### PR 6: Binding, Likelihood, and Parametric Fit Core

Goal:

- Add strict point-estimate fitting on top of the new graph and distribution stack.

Functional target:

- likelihood evaluation works from an explicit `State`.
- point-estimate fitting returns a result object instead of mutating the model.
- a fitted result can be bound back to the model for inspection/evaluation.
- at least one non-trivial composed model, such as `Bernoulli(p=sigmoid(linear()).with_covariate(x))`, fits end to end.

Touch:

- `pykelihood/bound.py`
- `pykelihood/likelihood.py`
- `pykelihood/parametric/__init__.py`
- `pykelihood/_inference_utils.py` if needed
- fit-focused tests

Do not touch:

- reparametrization
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- fitting is an explicit operation returning `FitResult`
- models are no longer mutated in place by fitting
- evaluation/binding should require explicit full `State` values rather than
  implicitly falling back to parameter init values; make that semantic change in
  this PR together with `bind`, not earlier in the migration

Test movement:

- add tests for:
  - `bind`
  - `log_likelihood`
  - `negative_log_likelihood`
  - `fit_mle`
  - fit-result binding
  - one small end-to-end composed model, such as Bernoulli with a sigmoid-transformed linear effect

Accept when:

- models can be evaluated against explicit `State`
- `fit_mle` has one strict score contract
- point-estimate results are separate objects rather than mutated distributions

### PR 7: Public API Cutover

Goal:

- Make the new graph/state/effect/distribution/parametric stack the default public API.

Functional target:

- the new execution path is the default public path.
- users no longer need the old `Parametrized` machinery for normal modeling and fitting.
- old tests that rely on mutation-based fitting are removed or rewritten.

Touch:

- `pykelihood/__init__.py`
- `pykelihood/parameters.py` as a temporary compatibility shim if needed
- `pykelihood/kernels.py` re-exports if needed
- remaining tests that still target the old execution path

Do not touch:

- reparametrization
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- the old `Parametrized` implementation is no longer the main execution path
- fitting moves from model methods to `fit_mle` and `FitResult`

Test movement:

- remove `tests/test_parameters.py`
- remove the remaining legacy distribution tests that still rely on old object mutation semantics

Accept when:

- importing `pykelihood` points to the new core
- the old execution path is gone from normal use
- temporary design notes are not part of the package state

### PR 8: Distribution Reparametrization Layer

Goal:

- Port reparametrization as a wrapper over existing `pykelihood` distributions.

Functional target:

- a plain `pykelihood` distribution can be wrapped to expose a different public parameter surface.
- value-level mappings such as `sigma -> exp(log_sigma)` are supported.
- fitting through that wrapper works.
- plain SciPy wrappers remain plain and do not grow renaming logic.

Touch:

- `pykelihood/reparametrization.py`
- reparametrization tests

Do not touch:

- plain SciPy wrapping behavior
- profiler
- metrics
- Bayesian code

Behavior intentionally changed:

- SciPy wrappers remain plain wrappers
- reparametrization becomes a separate wrapper layer over `pykelihood` distributions

Test movement:

- replace `tests/test_parametrization.py` with tests for:
  - renamed public parameters
  - fixed values
  - value-level derived parameter mappings
  - reparametrized fitting over plain `pykelihood` distributions

Accept when:

- reparametrized distributions work on the new core
- plain SciPy wrappers stay simple
- reparametrization is no longer coupled to the old parameter container logic

### PR 9: Parametric Profiler

Goal:

- Rebuild profile likelihood tooling as part of the parametric layer.

Functional target:

- profile likelihood sweeps work on top of explicit states and refits.
- confidence intervals come from the profiler/result layer, not from mutable fitted distributions.
- the score contract is strict and singular.

Touch:

- `pykelihood/parametric/profiler.py`
- profiler tests

Do not touch:

- metrics
- Bayesian code

Behavior intentionally changed:

- profiling works through explicit state updates and refits
- confidence intervals move off fitted distribution objects and onto the profiler/result layer

Test movement:

- replace `tests/test_profiler.py` with tests for profile sweeps, optima, and confidence intervals

Accept when:

- profiler no longer probes multiple score signatures
- confidence intervals are computed from the new fit objects
- the old fitted-distribution profiling API is gone

### PR 10: Metrics Port

Goal:

- Move the metrics layer onto the new distribution and explicit-state contracts.

Functional target:

- metrics operate on the new distribution surface.
- they do not require the old fitted-distribution conventions.
- they can be tested independently of legacy parameter plumbing.

Touch:

- `pykelihood/metrics.py`
- metrics tests

Do not touch:

- Bayesian code

Behavior intentionally changed:

- metrics become plain functions over the new distribution surface and fit results

Test movement:

- replace `tests/test_metrics.py` with tests for likelihood-based and scoring-rule metrics on the new surface

Accept when:

- metrics depend only on the new distribution surface
- no legacy parameter plumbing remains in metrics

### PR 11: Bayesian Module

Goal:

- Add Bayesian fitting only after the parametric core is stable.

Functional target:

- MAP and MCMC exist as separate fitting modes.
- posterior results are separate from ordinary fit results.
- the parametric core does not need to know Bayesian internals to keep working.

Touch:

- `pykelihood/bayesian.py`
- `pykelihood/__init__.py`
- Bayesian tests

Do not touch:

- optional backend integration

Behavior intentionally changed:

- `fit_map`, `fit_mcmc`, and `PosteriorResult` become available as separate additions to the modeling layer

Test movement:

- add tests for MAP fitting, posterior traces, posterior-state helpers, and predictive sampling

Accept when:

- Bayesian fitting exists without changing the parametric core semantics
- posterior results stay separate from ordinary fitted-state results

### PR 12: Optional Backend Integration

Goal:

- Add optional backend execution after the Bayesian API is already stable.

Functional target:

- optional backends can be used without becoming mandatory dependencies.
- backend dispatch is declarative and lazy.
- no non-Bayesian behavior changes in this PR.

Touch:

- `pykelihood/backends/__init__.py`
- `pykelihood/backends/pymc.py`
- packaging metadata for optional dependencies
- backend-specific tests

Do not touch:

- deterministic modeling semantics
- profiler
- metrics

Behavior intentionally changed:

- Bayesian fitting can dispatch to optional backends while `import pykelihood` still works without those dependencies

Test movement:

- add backend tests for lazy import, backend dispatch, and declarative distribution mapping

Accept when:

- PyMC is optional
- backend imports are lazy
- backend mapping is declarative rather than hand-written per call site

### PR 13: Compatibility Cleanup

Goal:

- Remove migration-only shims and stale names.

Functional target:

- any remaining shim exists only if it still has real compatibility value.
- the surviving module layout reads like the redesign, not like a transition scaffold.

Touch:

- `pykelihood/parameters.py` if it still exists as a shim
- `pykelihood/kernels.py` if it no longer adds compatibility value
- temporary re-exports
- docs or tests that still refer to migration-only import paths

Do not touch:

- core behavior

Behavior intentionally changed:

- the remaining package layout reflects the target architecture rather than the migration path

Accept when:

- no migration-only shim remains unless it still serves a real compatibility purpose
- the codebase reads like the redesign rather than a layered transition

## Not Part Of This Migration

The following are real future topics, but they should not widen the migration PRs:

- partial binding as a first-class feature distinct from full binding
- richer regression and categorical effect builders
- generic transformed distributions beyond the current wrapper taxonomy
- posterior/predictive object refactors beyond the current Bayesian result layer
- non-parametric model support

Those should be handled after the redesigned core lands.
