from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy import stats

from pykelihood import parameters
from pykelihood.distributions import Normal
from pykelihood.distributions._compat import CompatibilityValue
from pykelihood.distributions.core import (
    Distribution,
    ParameterDefault,
    ParameterInput,
    ScipyDistribution,
)
from pykelihood.effects import linear
from pykelihood.likelihood import log_likelihood, negative_log_likelihood
from pykelihood.parametric import FitResult, Objective, fit_mle
from pykelihood.state import ParameterLayout, PositiveTransform, initial_state


class BareParametrized(parameters.Parametrized):
    def __init__(self, *params: parameters.Parametrized | float, names: list[str]):
        super().__init__(*params)
        self._params_names = tuple(names)

    @property
    def params_names(self):
        return self._params_names


def test_parameter_layout_traversal_is_deterministic_and_shares_parameters() -> None:
    alpha = parameters.Parameter(2.0)
    beta = parameters.Parameter(3.0)
    expr = (alpha + beta) + alpha

    layout = ParameterLayout.from_expr(expr)

    assert layout.parameters == (alpha, beta)
    assert layout.parameter_paths == {
        alpha: (("left", "left"), ("right",)),
        beta: (("left", "right"),),
    }


def test_array_valued_parameter_flattens_and_unflattens() -> None:
    alpha = parameters.Parameter(np.array([1.0, 2.0]))
    beta = parameters.Parameter(3.0)
    expr = BareParametrized(alpha, beta, names=["alpha", "beta"])
    layout = ParameterLayout.from_expr(expr)

    state = initial_state(expr)
    np.testing.assert_allclose(layout.flatten(state), np.array([1.0, 2.0, 3.0]))

    updated = layout.unflatten(np.array([4.0, 5.0, 6.0]))

    np.testing.assert_allclose(updated[alpha], np.array([4.0, 5.0]))
    np.testing.assert_allclose(updated[beta], np.array([6.0]))


def test_two_dimensional_parameter_flattens_and_unflattens() -> None:
    alpha = parameters.Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    beta = parameters.Parameter(5.0)
    expr = BareParametrized(alpha, beta, names=["alpha", "beta"])
    layout = ParameterLayout.from_expr(expr)

    state = initial_state(expr)
    np.testing.assert_allclose(
        layout.flatten(state), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )

    updated = layout.unflatten(np.array([6.0, 7.0, 8.0, 9.0, 10.0]))

    np.testing.assert_allclose(updated[alpha], np.array([[6.0, 7.0], [8.0, 9.0]]))
    np.testing.assert_allclose(updated[beta], np.array([10.0]))


def test_parameter_layout_applies_transforms_during_flatten_and_unflatten() -> None:
    scale = parameters.Parameter(2.0, transform=PositiveTransform())
    layout = ParameterLayout.from_expr(scale)
    state = {scale: np.array(2.0)}

    np.testing.assert_allclose(layout.flatten(state), np.array([2.0]))
    np.testing.assert_allclose(
        layout.flatten(state, transform=True), np.array([np.log(2.0)])
    )

    recovered = layout.unflatten(np.array([np.log(2.0)]), transform=True)

    np.testing.assert_allclose(recovered[scale], np.array([2.0]))


def test_parameter_layout_applies_transforms_to_non_scalar_parameters() -> None:
    scale = parameters.Parameter(np.array([2.0, 3.0]), transform=PositiveTransform())
    layout = ParameterLayout.from_expr(scale)
    state = {scale: np.array([2.0, 3.0])}

    np.testing.assert_allclose(layout.flatten(state), np.array([2.0, 3.0]))
    np.testing.assert_allclose(
        layout.flatten(state, transform=True), np.log(np.array([2.0, 3.0]))
    )

    recovered = layout.unflatten(np.log(np.array([2.0, 3.0])), transform=True)

    np.testing.assert_allclose(recovered[scale], np.array([2.0, 3.0]))


def test_state_values_take_precedence_over_parameter_init() -> None:
    alpha = parameters.Parameter(1.0)
    layout = ParameterLayout.from_expr(alpha)
    state = {alpha: np.array(5.0)}

    np.testing.assert_allclose(layout.flatten(state), np.array([5.0]))


def test_parameter_layout_initial_state_requires_initial_values() -> None:
    alpha = parameters.Parameter(shape=(2,))
    with pytest.raises(ValueError, match="uninitialized"):
        initial_state(alpha)


def test_state_indexing() -> None:
    alpha = parameters.Parameter(1.0)
    state = initial_state(alpha)

    assert isinstance(state, dict)
    assert len(state) == 1
    np.testing.assert_allclose(state[alpha], np.array(1.0))


def _normal(
    loc: ParameterInput = None, scale: ParameterInput = None
) -> ScipyDistribution:
    return ScipyDistribution(
        stats.norm,
        {"loc": loc, "scale": scale},
        defaults={
            "loc": ParameterDefault(0.0),
            "scale": ParameterDefault(1.0, PositiveTransform()),
        },
    )


def test_likelihood_functions_sum_explicit_state_logpdf_values() -> None:
    loc = parameters.Parameter(0.0)
    scale = parameters.Parameter(1.0, transform=PositiveTransform())
    model = _normal(loc, scale)
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}
    data = np.asarray([1.0, 2.0, 3.0])

    expected = np.sum(stats.norm.logpdf(data, loc=2.0, scale=3.0))

    assert log_likelihood(model, data, state=state) == pytest.approx(expected)
    assert negative_log_likelihood(model, data, state=state) == pytest.approx(-expected)


def test_fit_mle_returns_owned_physical_state_and_optimizer_metadata() -> None:
    model = _normal()
    loc = model.parameters["loc"]
    scale = model.parameters["scale"]
    assert isinstance(loc, parameters.Parameter)
    assert isinstance(scale, parameters.Parameter)
    input_state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}
    original_state = {
        parameter: value.copy() for parameter, value in input_state.items()
    }

    result = fit_mle(model, np.asarray([1.0, 2.0, 3.0]), state=input_state)

    assert isinstance(result, FitResult)
    assert result.model is model
    assert result.optimizer_layout.parameters == (loc, scale)
    np.testing.assert_allclose(result.optimizer_x0, [2.0, np.log(3.0)])
    np.testing.assert_allclose(result.optimizer_x, result.optimize_result.x)
    for parameter, value in original_state.items():
        np.testing.assert_array_equal(input_state[parameter], value)
        assert result.state[parameter] is not input_state[parameter]


def test_fit_mle_uses_transformed_coordinates_and_identity_fixed_refits() -> None:
    loc = parameters.Parameter(0.0)
    scale = parameters.Parameter(1.0, transform=PositiveTransform())
    model = _normal(loc, scale)
    data = np.asarray([-1.0, 0.0, 1.0])

    result = fit_mle(model, data, x0=[0.0, 2.0])
    refit = fit_mle(model, data, state=result.state, fixed={loc: 2.0})

    np.testing.assert_allclose(result.optimizer_x0, [0.0, np.log(2.0)])
    assert loc not in refit.optimizer_layout.parameters
    assert refit.optimizer_layout.parameters == (scale,)
    np.testing.assert_allclose(refit.state[loc], 2.0)
    assert refit.fixed[loc] is not result.state[loc]


def test_fit_mle_supports_bound_effect_parameters() -> None:
    slope = parameters.Parameter(0.5)
    covariate = np.asarray([0.0, 1.0, 2.0])
    model = _normal(loc=linear(slope=slope).with_covariate(covariate), scale=1.0)

    result = fit_mle(model, covariate)

    assert result.optimizer_layout.parameters == (slope,)
    assert result.state[slope] == pytest.approx(1.0, abs=0.1)


def test_fit_mle_handles_zero_free_parameters() -> None:
    model = _normal(loc=0.0, scale=1.0)
    data = np.asarray([0.0, 1.0])

    result = fit_mle(model, data)

    assert result.optimizer_layout.vector_size == 0
    assert result.optimize_result.success
    assert result.optimize_result.nfev == 1
    assert result.state == {}


def test_fit_mle_all_fixed_parameters_skips_minimize(monkeypatch) -> None:
    loc = parameters.Parameter(0.0)
    scale = parameters.Parameter(1.0, transform=PositiveTransform())
    model = _normal(loc, scale)

    def unexpected_minimize(*args, **kwargs):
        pytest.fail("minimize must not run when all parameters are fixed")

    monkeypatch.setattr("pykelihood.parametric.fitting.minimize", unexpected_minimize)
    result = fit_mle(
        model,
        np.asarray([0.0, 1.0]),
        fixed={loc: np.asarray(2.0), scale: np.asarray(3.0)},
    )

    assert result.optimizer_layout.vector_size == 0
    np.testing.assert_allclose(result.state[loc], 2.0)
    np.testing.assert_allclose(result.state[scale], 3.0)


def test_fit_mle_allows_supplied_values_for_uninitialized_parameters() -> None:
    loc = parameters.Parameter()
    model = _normal(loc=loc, scale=1.0)

    result = fit_mle(model, np.asarray([0.0, 1.0]), state={loc: np.asarray(2.0)})
    np.testing.assert_allclose(result.optimizer_x0, [2.0])

    fixed_result = fit_mle(model, np.asarray([0.0, 1.0]), fixed={loc: np.asarray(2.0)})
    np.testing.assert_allclose(fixed_result.state[loc], 2.0)


def test_fit_mle_deduplicates_shared_fixed_parameters() -> None:
    shared = parameters.Parameter(1.0, transform=PositiveTransform())
    model = _normal(loc=shared, scale=shared)

    result = fit_mle(model, np.asarray([0.0, 1.0]), fixed={shared: 2.0})

    assert result.optimizer_layout.parameters == ()
    assert tuple(result.state) == (shared,)
    assert tuple(result.fixed) == (shared,)


def test_fit_mle_validates_fixed_shapes_and_owns_values() -> None:
    loc = parameters.Parameter(np.zeros(2))
    model = _normal(loc=loc, scale=1.0)
    fixed_value = np.asarray([1.0, 2.0])

    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        fit_mle(model, np.asarray([0.0, 1.0]), fixed={loc: np.asarray(1.0)})

    result = fit_mle(model, np.asarray([0.0, 1.0]), fixed={loc: fixed_value})
    fixed_value[:] = 9.0
    np.testing.assert_allclose(result.fixed[loc], [1.0, 2.0])
    np.testing.assert_allclose(result.state[loc], [1.0, 2.0])


def test_fit_mle_converts_optimizer_coordinates_to_physical_state() -> None:
    loc = parameters.Parameter(0.0)
    scale = parameters.Parameter(1.0, transform=PositiveTransform())
    model = _normal(loc, scale)

    result = fit_mle(
        model,
        np.asarray([-1.0, 0.0, 1.0]),
        x0=[0.0, 2.0],
        scipy_args={"options": {"maxiter": 1}},
    )

    np.testing.assert_allclose(result.state[loc], result.optimizer_x[0])
    np.testing.assert_allclose(result.state[scale], np.exp(result.optimizer_x[1]))


def test_fit_mle_rejects_unknown_state_and_wrong_x0_length() -> None:
    model = _normal()
    data = np.asarray([0.0, 1.0])
    unknown = parameters.Parameter(0.0)

    with pytest.raises(ValueError, match="state contains parameters"):
        fit_mle(model, data, state={unknown: np.asarray(0.0)})
    with pytest.raises(ValueError, match="Expected 2 values in x0"):
        fit_mle(model, data, x0=[0.0])


def test_fit_mle_rejects_non_scalar_objective() -> None:
    model = _normal(loc=0.0, scale=1.0)

    def objective(model, data, *, state):
        return np.asarray([1.0, 2.0])

    with pytest.raises(TypeError, match="must return one scalar"):
        fit_mle(model, np.asarray([0.0]), objective=cast(Objective, objective))


def test_fit_mle_rejects_nan_but_preserves_infinite_objectives() -> None:
    model = _normal(loc=0.0, scale=1.0)

    def nan_objective(model, data, *, state):
        return np.nan

    def infinite_objective(model, data, *, state):
        return np.inf

    with pytest.raises(ValueError, match="returned NaN"):
        fit_mle(model, np.asarray([0.0]), objective=nan_objective)

    result = fit_mle(model, np.asarray([0.0]), objective=infinite_objective)
    assert np.isinf(result.optimize_result.fun)


def test_fit_mle_validates_transformed_parameter_domains() -> None:
    scale = parameters.Parameter(1.0, transform=PositiveTransform())
    model = _normal(scale=scale)

    with pytest.raises(ValueError, match="outside its transform domain"):
        fit_mle(model, np.asarray([0.0]), state={scale: np.asarray(0.0)})


def test_compatibility_fit_result_binds_state_and_refits_named_parameters() -> None:
    model = Normal()
    loc = model.parameters["loc"]
    assert isinstance(loc, parameters.Parameter)
    data = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    result = model.fit(data, method="Powell")
    fitted = cast(Distribution, result)

    np.testing.assert_allclose(
        fitted.logpdf(data), model.logpdf(data, state=result.state)
    )
    assert result.model is model
    loc_projection = cast(CompatibilityValue, result.flattened_param_dict["loc"])
    assert loc_projection.value == pytest.approx(result.state[loc])
    assert loc_projection() == pytest.approx(result.state[loc])

    refit = result.fit(loc=3.0)

    assert refit.model is model
    assert loc in refit.fixed
    assert cast(CompatibilityValue, refit.loc).value == pytest.approx(3.0)


def test_core_fit_result_refit_preserves_original_and_existing_fixed_parameters() -> (
    None
):
    model = _normal()
    loc = model.parameters["loc"]
    scale = model.parameters["scale"]
    assert isinstance(loc, parameters.Parameter)
    assert isinstance(scale, parameters.Parameter)
    data = np.asarray([1.0, 2.0, 3.0, 4.0])

    result = fit_mle(
        model,
        data,
        state={loc: np.asarray(1.0), scale: np.asarray(2.0)},
        fixed={scale: 2.0},
    )
    original_state = {
        parameter: value.copy() for parameter, value in result.state.items()
    }
    original_fixed = {
        parameter: value.copy() for parameter, value in result.fixed.items()
    }

    refit = result.fit(data, loc=3.0, scipy_args={"method": "Powell"})

    assert refit.model is model
    np.testing.assert_allclose(refit.state[loc], 3.0)
    np.testing.assert_allclose(refit.state[scale], 2.0)
    assert set(refit.fixed) == {loc, scale}
    for parameter, value in original_state.items():
        np.testing.assert_array_equal(result.state[parameter], value)
    for parameter, value in original_fixed.items():
        np.testing.assert_array_equal(result.fixed[parameter], value)


def test_core_fit_result_refit_resolves_nested_shared_leaf_names() -> None:
    shared = parameters.Parameter(1.0)
    covariate = np.asarray([1.0, 2.0, 3.0])
    model = _normal(loc=linear(slope=shared).with_covariate(covariate), scale=shared)
    result = fit_mle(model, covariate, state={shared: np.asarray(1.0)})

    mapping = result.param_mapping()
    assert mapping == [(pytest.approx(result.state[shared]), ("loc_slope", "scale"))]

    refit = result.fit(covariate, loc_slope=2.0)

    assert tuple(refit.fixed) == (shared,)
    np.testing.assert_allclose(refit.state[shared], 2.0)
    with pytest.raises(ValueError, match="structural.*leaf Parameter"):
        result.fit(covariate, loc=2.0)


def test_legacy_compatibility_score_is_minimized() -> None:
    data = np.asarray([1.0, 2.0, 3.0])

    def score(distribution, values) -> float:
        return -float(np.sum(distribution.logpdf(values)))

    result = Normal().fit(data, score=score, method="Powell")

    assert result.optimize_result.fun == pytest.approx(score(result, data))
