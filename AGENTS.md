# AGENTS.md

## Project Context

`pykelihood` is a Python library for likelihood-based statistical modeling. It extends the practical modeling surface around `scipy.stats` by supporting object-oriented distributions, parameter trends/effects, custom likelihood penalties, reparametrized distributions, point-estimate fitting, metrics, and profile-likelihood tooling.

The profiler is especially important. It is one of the most advanced and useful parts of the library, and must keep working throughout the migration.

## Migration Goal

The project is in a large internal rewrite from the old `Parametrized`-centered architecture to a new architecture based on:

- expression graph nodes in `pykelihood/expr.py`
- explicit parameter state/layout in `pykelihood/state.py`
- covariate-dependent effects in `pykelihood/effects.py`
- plain SciPy distribution wrappers plus separate reparametrization
- point-estimate fitting in `pykelihood/parametric/`
- Bayesian fitting as a later layer over the same model graph

The key architectural goal is to stop using the old `Parametrized` hierarchy as the execution model. The key product goal is different: keep existing public APIs alive for the next release wherever practical.

## Compatibility Policy

For the next release, preserve compatibility surfaces as deprecated shims/projections when they can be implemented honestly on top of the new model. Do not remove useful downstream APIs just because they are no longer core concepts.

Important compatibility surfaces include:

- `Distribution.fit(...)` as a forwarder to `fit_mle`
- fit-result `.fit(...)` for profiler refits
- `params_names`
- `flattened_params`
- `optimisation_params`
- `optimisation_param_dict`
- `flattened_param_dict`
- `param_mapping()`
- `pykelihood.parameters` as a compatibility import path
- `pykelihood.kernels` as a compatibility layer over effects
- `pykelihood.profiler.Profiler`

These should delegate to the new graph/state/fitting implementation. Do not resurrect old internals to satisfy them.

The next breaking release can remove or simplify deprecated compatibility APIs. Until then, keep them working and tested.

## Migration Steps

Use `MIGRATION_PLAN.md` as the source of truth. In short:

1. Node core, explicit state/layout, and effects are already landed.
2. Finish the distribution and `fit_mle` core while preserving deprecated accessors needed downstream.
3. Cut public imports over to the new stack without breaking the profiler.
4. Move reparametrization into a separate wrapper layer.
5. Rebuild the profiler on explicit states and fixed-parameter refits, while keeping the old import path as a delegating wrapper.
6. Port metrics onto `FitResult` and explicit-state distributions.
7. Add Bayesian fitting and optional PyMC backend integration after the parametric core is stable.
8. In a later compatibility-cleanup/breaking release, remove migration-only shims and deprecated APIs that no longer need to survive.

When in doubt, prefer preserving user-facing behavior with a thin deprecated adapter over deleting it during this migration.
