import numpy as np
import numpy.testing
from numpy.typing import ArrayLike

from pykelihood import kernels
from pykelihood.distributions import Uniform
from pykelihood.metrics import log_likelihood


def uniform_bounds(params: dict[str, ArrayLike]) -> dict[str, float]:
    return {"loc": params["a"], "scale": params["b"] - params["a"]}


def test_reparametrization_names():
    u = Uniform(a=0, b=1, reparametrization=uniform_bounds)
    assert u.params_names == ("a", "b")


def test_reparametrization_rvs():
    u = Uniform(a=3, b=5, reparametrization=uniform_bounds)
    sample = u.rvs(10000)
    numpy.testing.assert_array_less(sample, 5)
    numpy.testing.assert_array_less(3, sample)


def test_reparametrization_pdf():
    u = Uniform(a=3, b=5, reparametrization=uniform_bounds)
    u_standard = Uniform(loc=3, scale=2)
    x = np.linspace(2, 6, 100)
    numpy.testing.assert_array_almost_equal(u.pdf(x), u_standard.pdf(x))


def test_reparametrization_kernel():
    x = np.linspace(2, 6, 100)
    u = Uniform(a=2, b=kernels.linear(x, a=2, b=1), reparametrization=uniform_bounds)
    u.pdf(x)
    u.rvs(x.size)


def test_reparametrization_with_params():
    u = Uniform(a=3, b=5, reparametrization=uniform_bounds)
    v = u.with_params(a=2, b=4)
    assert v.a() == 2
    assert v.b() == 4


def test_reparametrization_fit():
    u = Uniform(a=0, b=5, reparametrization=uniform_bounds)
    data = np.random.uniform(1, 3, 1000)
    fit = u.fit(data)
    assert 0 < fit.a() <= 1.1
    assert 2.9 <= fit.b() < 5
    assert log_likelihood(fit, data) > log_likelihood(u, data)


def test_reparametrization_fit_fixed_param():
    u = Uniform(a=0, b=5, reparametrization=uniform_bounds)
    data = np.random.uniform(1, 3, 1000)
    fit = u.fit(data, a=0.5)
    assert fit.a() == 0.5
    assert 2.9 <= fit.b() < 5
    assert log_likelihood(fit, data) > log_likelihood(u, data)


def test_reparametrization_fit_kernel():
    u = Uniform(a=0, b=kernels.constant(value=5), reparametrization=uniform_bounds)
    data = np.random.uniform(1, 3, 1000)
    fit = u.fit(data, a=0.5)
    assert fit.a() == 0.5
    assert 2.9 <= fit.b() < 5
    assert log_likelihood(fit, data) > log_likelihood(u, data)
