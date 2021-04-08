import numpy as np
import pytest
from scipy.stats import genextreme

from pykelihood import kernels
from pykelihood.distributions import GEV, Normal, TruncatedDistribution

REL_PREC = 1e-7
ABS_PREC = 0.1


def approx(x):
    return pytest.approx(x, rel=REL_PREC, abs=ABS_PREC)


@pytest.fixture
def linear_kernel(dataset):
    return kernels.linear(np.arange(len(dataset)))


class TestGEV:
    def test_fit(self, datasets):
        for ds in datasets:
            c, loc, scale = genextreme.fit(ds)
            fit = GEV.fit(ds)
            assert fit.loc == approx(loc)
            assert fit.scale == approx(scale)
            assert fit.shape == approx(-c)

    def test_fixed_values(self):
        data = np.random.standard_normal(1000)
        raw = Normal.fit(data)
        assert raw.loc == approx(0.0)
        assert raw.scale == approx(1.0)
        fixed = Normal.fit(data, loc=1.0)
        assert fixed.loc == 1.0


def test_cache():
    n = Normal(0, 1)
    np.testing.assert_array_almost_equal(
        n.pdf([-1, 0, 1]), [0.24197072, 0.39894228, 0.24197072]
    )
    np.testing.assert_array_almost_equal(
        n.pdf([1, 2, 3]), [0.24197072, 0.05399097, 0.00443185]
    )
    np.testing.assert_array_almost_equal(
        n.cdf([-1, 0, 1]), [0.15865525, 0.5, 0.84134475]
    )
    np.testing.assert_array_almost_equal(
        n.pdf([-1, 0, 1]), [0.24197072, 0.39894228, 0.24197072]
    )


def test_basic_with_params():
    n = Normal()
    m = n.with_params([2, 3])
    assert m.loc == 2
    assert m.scale == 3


def test_named_with_params():
    n = Normal()
    m = n.with_params(loc=2, scale=3)
    assert m.loc == 2
    assert m.scale == 3


def test_named_with_params_multi_level(linear_kernel):
    n = Normal(loc=linear_kernel, scale=1)
    m = n.with_params(loc_a=2, scale=3)
    assert m.loc.a == 2
    assert m.scale == 3


def test_named_with_params_partial_assignment():
    n = Normal()
    m = n.with_params(scale=3)
    assert m.loc == 0
    assert m.scale == 3


def test_fit_instance(dataset):
    std_fit = Normal.fit(dataset)
    instance_fit = Normal(loc=kernels.constant()).fit_instance(dataset)
    assert std_fit.loc() == approx(instance_fit.loc())


def test_rvs():
    n = Normal(1)
    sample = n.rvs(10000, scale=2)
    assert np.mean(sample) == approx(1)
    assert np.std(sample) == approx(2)


def test_rvs_random_state():
    n = Normal()
    rand_state = 10
    sample = n.rvs(10000, random_state=rand_state)
    sample2 = n.rvs(10000, random_state=rand_state)
    assert (sample == sample2).all()


def test_truncated_distribution_cdf():
    n = Normal()
    truncated = TruncatedDistribution(Normal(), lower_bound=0)
    assert truncated.cdf(-1) == 0
    assert truncated.cdf(0) == 0
    assert truncated.cdf(1) == 2 * (n.cdf(1) - n.cdf(0))
    assert truncated.cdf(np.inf) == 1


def test_truncated_distribution_fit():
    n = Normal(2)
    data = n.rvs(10000)
    trunc_data = data[data >= 0]
    truncated = TruncatedDistribution(Normal(), lower_bound=0)
    fitted_all_data = truncated.fit_instance(data)
    fitted_trunc = truncated.fit_instance(trunc_data)
    assert fitted_trunc.flattened_params == approx(fitted_all_data.flattened_params)
