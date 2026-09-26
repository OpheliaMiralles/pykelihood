import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from pykelihood.distributions.extreme_value import GEV, GPD
from pykelihood.expr import Constant
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.parametric import fit_mle
from pykelihood.state import ParameterLayout, PositiveTransform


def test_extreme_value_defaults_are_free_and_shape_is_publicly_visible() -> None:
    gev = GEV()
    gpd = GPD()

    for model in (gev, gpd):
        assert isinstance(model.parameters["loc"], Parameter)
        assert isinstance(model.parameters["scale"], Parameter)
        assert isinstance(model.parameters["scale"].transform, PositiveTransform)
        assert isinstance(model.parameters["shape"], Parameter)
        assert model.parameters["shape"].init == 0.0
        assert ParameterLayout.from_expr(model).parameters == tuple(
            model.parameters.values()
        )


def test_gev_uses_negative_scipy_shape_and_keeps_literals_constant() -> None:
    model = GEV(loc=1.0, scale=2.0, shape=0.25)
    values = np.array([-1.0, 0.0, 1.0, 2.0])

    assert all(
        isinstance(model.parameters[name], Constant) for name in model.parameters
    )
    assert_allclose(
        model.pdf(values), stats.genextreme.pdf(values, c=-0.25, loc=1.0, scale=2.0)
    )
    assert_allclose(
        model.cdf(values), stats.genextreme.cdf(values, c=-0.25, loc=1.0, scale=2.0)
    )


def test_gpd_uses_positive_scipy_shape_and_explicit_state() -> None:
    shape = Parameter(init=0.0)
    model = GPD(loc=0.5, scale=2.0, shape=shape)
    state = {shape: np.asarray(0.3)}
    values = np.array([0.5, 1.0, 2.0, 4.0])

    assert model.parameters["shape"] is shape
    assert ParameterLayout.from_expr(model).parameters == (shape,)
    assert_allclose(
        model.pdf(values, state=state),
        stats.genpareto.pdf(values, c=0.3, loc=0.5, scale=2.0),
    )
    assert_allclose(
        model.ppf([0.1, 0.5, 0.9], state=state),
        stats.genpareto.ppf([0.1, 0.5, 0.9], c=0.3, loc=0.5, scale=2.0),
    )


def test_gpd_shape_can_be_fitted_without_mutating_the_model() -> None:
    model = GPD(loc=0.0, scale=1.0)
    shape = model.parameters["shape"]
    assert isinstance(shape, Parameter)
    data = stats.genpareto.ppf(np.linspace(0.05, 0.95, 19), c=0.25)
    initial_score = negative_log_likelihood(model, data)

    result = fit_mle(model, data)

    assert result.model is model
    assert result.optimize_result.success
    assert tuple(result.state) == (shape,)
    assert np.isfinite(result.state[shape])
    assert negative_log_likelihood(model, data, state=result.state) < initial_score
    assert shape.init == 0.0
