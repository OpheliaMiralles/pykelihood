# PR 6: Public API Cutover — Explicit Implementation Plan

## Goal

Make the new graph/state/effect/distribution/parametric stack the default public API. The old `Parametrized` hierarchy becomes a compatibility shim, not the main execution path.

---

## Decision Summary

| Aspect | Decision |
|--------|----------|
| `pykelihood/parameters.py` | Thin compatibility shim re-exporting `Parameter`, `Constant`, `State`, `ParameterLayout`, and transforms from `expr.py`/`state.py`. Old `Parametrized`, `ConstantParameter`, `ParametrizedFunction` stay here as legacy internals, not public API. |
| `pykelihood/kernels.py` | Keep as compatibility layer over `effects.py`. No removal. |
| `pykelihood/__init__.py` | Minimal surface: only the new core (`distributions`, `parametric`, `effects`, `kernels`, `metrics`, `profiler`, `likelihood`). Do NOT re-export legacy `parameters` internals. |

---

## Step-by-Step Changes

### 1. Fix the IndentationError in `pykelihood/distributions/base.py`

**File:** `pykelihood/distributions/base.py`
**Lines:** 145–204

The `with_state` method is followed by orphaned code (lines 156–203) that should be inside a `fit` method. This is currently blocking all test collection.

**Action:** Remove the orphaned block entirely. The real fitting logic now lives in `pykelihood/parametric/fitting.py` as `fit_mle`. Any compatibility `.fit(...)` forwarder should be added explicitly as a new method, not as leftover indented debris.

### 2. Add `Distribution.fit(...)` compatibility forwarder

**File:** `pykelihood/distributions/base.py`
**Add after `with_state`:**

```python
def fit(self, data, **kwargs):
    from pykelihood.parametric.fitting import fit_mle
    return fit_mle(self, data, **kwargs)
```

This preserves the old mutation-free compatibility surface without resurrecting the old fitting implementation.

### 3. Update `pykelihood/__init__.py`

**File:** `pykelihood/__init__.py`

```python
from pykelihood import distributions  # noqa: F401
from pykelihood import effects  # noqa: F401
from pykelihood import kernels  # noqa: F401
from pykelihood import likelihood  # noqa: F401
from pykelihood import metrics  # noqa: F401
from pykelihood import parametric  # noqa: F401
from pykelihood import profiler  # noqa: F401

__all__ = [
    "distributions",
    "effects",
    "kernels",
    "likelihood",
    "metrics",
    "parametric",
    "profiler",
]
```

Minimal surface. Users who need low-level graph/state primitives import from `pykelihood.expr` and `pykelihood.state` explicitly.

### 4. Convert `pykelihood/parameters.py` to thin shim

**File:** `pykelihood/parameters.py`

**Keep public:**
- `Parameter` → re-export from current module (it already lives here, just ensure it's the node-based `Parameter`)
- `Constant` → re-export from `pykelihood.expr`
- `State` → re-export from `pykelihood.state`
- `ParameterLayout` → re-export from `pykelihood.state`
- `Transform`, `PositiveTransform`, `ProbabilityTransform` → re-exports from `pykelihood.state`

**Keep internal (not exported, but remain for compatibility):**
- `Parametrized`
- `ConstantParameter`
- `ParametrizedFunction`
- `ensure_parametrized`

**Implementation:** Add re-exports at the top of the file:

```python
from pykelihood.expr import Constant  # noqa: F401
from pykelihood.state import ParameterLayout, State, Transform  # noqa: F401
from pykelihood.state import PositiveTransform, ProbabilityTransform  # noqa: F401

__all__ = [
    "Parameter",
    "Constant",
    "State",
    "ParameterLayout",
    "Transform",
    "PositiveTransform",
    "ProbabilityTransform",
]
```

Old `Parametrized` machinery stays in this file but is **not** part of `__all__`. It exists only so internal code and any undocumented user imports don't break immediately.

### 5. Remove `tests/test_parameters.py`

**File:** `tests/test_parameters.py` → **DELETE**

All tests in this file target the old `Parametrized` API:
- `test_parameter` / `test_parameter_with_params` → covered by `tests/test_foundations.py` (node-based `Parameter` already tested)
- `test_flattened_params*` → covered by `tests/test_inference.py` (`ParameterLayout` and `initial_state` already tested)
- `TestParametrizedFunction` → the `ParametrizedFunction` class is being retired from the public path. Delete these tests.

Do **not** port `ParametrizedFunction` tests. The migration plan explicitly lists `ParametrizedFunction` as a cleanup candidate, not a migration target.

### 6. Convert `tests/test_kernels.py` to compatibility tests

**File:** `tests/test_kernels.py`

Since `kernels.py` is intentionally kept as a compatibility layer, this test file should be rewritten to validate that the **compatibility surface works**, not that the old kernel internals are correct.

**New test strategy:**
- Import `pykelihood.kernels` as the public path.
- Test that `kernels.linear`, `kernels.polynomial`, `kernels.linear_regression`, etc. still produce `Kernel` objects.
- Test that `Kernel.__call__()` still works.
- Test that `with_covariate` still works.
- **Remove** tests that assert old internal structure:
  - `len(trend.params)` / `len(trend.optimisation_params)` (these are `Parametrized` concepts)
  - `trend.with_params([3, 4])()` (old mutation API)
  - `flattened_param_dict` keys like `"y_x"`, `"b_a"` (old naming convention)
  - `param_mapping` structure

**Example rewrite:**

```python
def test_linear_kernel_evaluates(dataset):
    trend = kernels.linear(dataset)
    assert callable(trend)
    assert trend().shape == dataset.shape

def test_linear_kernel_with_covariate(dataset):
    trend = kernels.linear(dataset)
    new_covariate = dataset + 1.0
    updated = trend.with_covariate(new_covariate)
    assert updated().shape == new_covariate.shape

def test_linear_regression_returns_kernel(matrix_data):
    regression = kernels.linear_regression(matrix_data)
    from pykelihood.kernels import Kernel
    assert isinstance(regression, Kernel)
```

### 7. Update `tests/test_distributions.py` to stop using old `.fit(...)` mutation semantics

**File:** `tests/test_distributions.py`

Current problematic lines:
- Line 50: `std_fit = Normal().fit(dataset)` → replace with `fit_mle(Normal(), dataset)`
- Line 51–52: `std_fit.loc.eval({})` → `std_fit.state[std_fit.model.loc]`
- Line 57: `n = Normal().fit(dataset, loc=5.0)` → `fit_mle(Normal(), dataset, loc=5.0)`

**Important:** Keep these as `fit_mle` tests in this PR. Moving them to `tests/test_inference.py` is fine, but the PR must not lose the fitting scenarios that already exist here.

Also update:
- Line 5: `from pykelihood import distributions, kernels` → can keep `distributions`, but `kernels` is only used indirectly via `test_normal_fit` if at all. Check if `kernels` import is still needed.
- Line 12: `from pykelihood.parameters import ConstantParameter, Parameter` → `ConstantParameter` is no longer new-API. If any test uses `ConstantParameter`, replace with `Constant` from `pykelihood.expr` or just a literal, depending on context.

### 8. Update `tests/test_metrics.py` if it uses old `.fit(...)`

**File:** `tests/test_metrics.py`

Line 38: `n = Normal(loc=kernels.linear(np.arange(len(ndata)))).fit(ndata)`
→ `from pykelihood.parametric import fit_mle`
→ `n = fit_mle(Normal(loc=kernels.linear(np.arange(len(ndata)))), ndata)`

### 9. Update `tests/test_parametrization.py` to use new API

**File:** `tests/test_parametrization.py`

Line 20, 29, 63: `.fit(data)` → `fit_mle(model, data)`

Also ensure any `ConstantParameter` usage is replaced. The new API uses `Parameter(..., transform=...)` for fixed-value parameters via the reparametrization layer (PR 7), but for now, if the test just needs a fixed value, pass it as a literal or use `fit_mle(..., param_name=value)`.

---

## Acceptance Criteria

1. `import pykelihood` points to the new core.
2. `tests/test_parameters.py` no longer exists.
3. `tests/test_kernels.py` contains compatibility tests only (no `Parametrized` internals asserted).
4. `pykelihood/__init__.py` has minimal surface.
5. `pykelihood/parameters.py` is a thin shim: re-exports + legacy internals hidden from `__all__`.
6. The old `Parametrized`-based fitting path is gone from normal use; `.fit(...)` on a `Distribution` is a thin forwarder to `fit_mle`.
7. All tests pass.
8. `tests/test_distributions.py`, `tests/test_metrics.py`, `tests/test_parametrization.py` use `fit_mle` explicitly.

---

## Out of Scope for PR 6

- **Do not** touch `tests/test_profiler.py` (that's PR 8).
- **Do not** touch `pykelihood/reparametrization.py` (doesn't exist yet; PR 7).
- **Do not** touch `pykelihood/bayesian.py` (PR 10).
- **Do not** remove `pykelihood/parameters.py` entirely — keep the shim.
- **Do not** remove `pykelihood/kernels.py` — keep the compat layer.
- **Do not** attempt to refactor `Distribution` internals beyond fixing the orphaned code and adding the forwarder.

---

## Test Execution Order (for the migrator)

1. Fix `base.py` IndentationError.
2. Add `.fit(...)` forwarder.
3. Update `__init__.py`.
4. Convert `parameters.py` to shim.
5. Run pytest to confirm nothing is broken yet.
6. Rewrite `tests/test_kernels.py`.
7. Update `tests/test_distributions.py` fitting calls.
8. Update `tests/test_metrics.py` fitting calls.
9. Update `tests/test_parametrization.py` fitting calls.
10. Delete `tests/test_parameters.py`.
11. Run pytest again. Fix any import errors.
12. Confirm `tests/test_foundations.py`, `tests/test_effects.py`, `tests/test_inference.py` still pass unchanged.
