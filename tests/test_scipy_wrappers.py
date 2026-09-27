import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pykelihood.distributions.core import ScipyDistribution
from pykelihood.distributions.scipy_wrappers import Gamma, Genextreme, Norm, Normal
from pykelihood.expr import Constant
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.parametric import fit_mle
from pykelihood.state import ParameterLayout, PositiveTransform


def test_factory_builds_named_classes_with_native_parameters_and_alias() -> None:
    assert Norm is Normal
    assert Normal.__name__ == "Norm"
    assert Gamma.__name__ == "Gamma"
    assert Genextreme.__name__ == "Genextreme"
    assert tuple(inspect.signature(Normal).parameters) == ("loc", "scale")
    assert tuple(inspect.signature(Gamma).parameters) == ("a", "loc", "scale")
    assert tuple(inspect.signature(Genextreme).parameters) == ("c", "loc", "scale")
    assert_allclose(Norm(1.0, 2.0).pdf(0.5), stats.norm.pdf(0.5, loc=1.0, scale=2.0))


def test_generated_wrapper_requires_shape_and_keeps_scipy_parameterization() -> None:
    with pytest.raises(TypeError, match="Missing required.*a"):
        Gamma()
    with pytest.raises(TypeError, match="Unexpected distribution parameter"):
        Gamma(a=2.0, typo=1.0)
    with pytest.raises(TypeError, match="supplied more than once"):
        Gamma(2.0, a=3.0)
    with pytest.raises(TypeError, match="at most"):
        Gamma(2.0, 0.0, 1.0, 4.0)

    shape = Parameter(init=2.0)
    distribution = Gamma(a=shape, loc=1.0, scale=3.0)

    assert isinstance(distribution, ScipyDistribution)
    assert distribution.parameters["a"] is shape
    assert isinstance(distribution.parameters["loc"], Constant)
    assert isinstance(distribution.parameters["scale"], Constant)
    assert_allclose(distribution.parameters["loc"].value, 1.0)
    assert_allclose(distribution.parameters["scale"].value, 3.0)
    assert_allclose(
        distribution.pdf([1.0, 2.0]),
        stats.gamma.pdf([1.0, 2.0], a=2.0, loc=1.0, scale=3.0),
    )


def test_generated_defaults_are_free_parameters() -> None:
    distribution = Norm()
    location = distribution.parameters["loc"]
    scale = distribution.parameters["scale"]

    assert isinstance(location, Parameter)
    assert isinstance(scale, Parameter)
    assert isinstance(scale.transform, PositiveTransform)
    assert ParameterLayout.from_expr(distribution).parameters == (location, scale)
    assert_allclose(distribution.pdf(np.array([0.0, 1.0])), stats.norm.pdf([0.0, 1.0]))


def test_generated_gamma_fits_its_native_shape_parameter() -> None:
    shape = Parameter(init=1.0, transform=PositiveTransform())
    distribution = Gamma(a=shape, loc=0.0, scale=1.0)
    data = stats.gamma.ppf(np.linspace(0.1, 0.9, 9), a=2.0)
    initial_score = negative_log_likelihood(distribution, data)

    result = fit_mle(distribution, data)

    assert result.model is distribution
    assert result.optimize_result.success
    assert tuple(result.state) == (shape,)
    expected_shape = stats.gamma.fit(data, floc=0.0, fscale=1.0)[0]
    assert result.state[shape] == pytest.approx(expected_shape, rel=1e-3)
    assert (
        negative_log_likelihood(distribution, data, state=result.state) < initial_score
    )
    assert shape.init is not None
    assert_allclose(shape.init, 1.0)
