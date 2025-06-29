import numpy as np
import pytest
from scipy import stats

from pykelihood import distributions, kernels
from pykelihood.distributions import (
    GEV,
    Normal,
    TruncatedDistribution,
    _name_from_scipy_dist,
)
from pykelihood.kernels import linear
from pykelihood.parameters import ConstantParameter, Parameter

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
            c, loc, scale = stats.genextreme.fit(ds)
            fit = GEV().fit(ds)
            assert fit.loc() == approx(loc)
            assert fit.scale() == approx(scale)
            assert fit.shape() == approx(-c)

    def test_fixed_values(self):
        data = np.random.standard_normal(1000)
        raw = Normal().fit(data)
        assert raw.loc() == approx(0.0)
        assert raw.scale() == approx(1.0)
        fixed = Normal().fit(data, loc=1.0)
        assert fixed.loc() == 1.0


def test_cache():
    """There is no cache anymore, the test is kept as it can still be useful."""
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
    assert m.loc() == 2
    assert m.scale() == 3


def test_named_with_params():
    n = Normal()
    m = n.with_params(loc=2, scale=3)
    assert m.loc() == 2
    assert m.scale() == 3


def test_named_with_params_multi_level(linear_kernel):
    n = Normal(loc=linear_kernel, scale=1)
    m = n.with_params(loc_a=2, scale=3)
    assert m.loc.a() == 2
    assert m.scale() == 3


def test_named_with_params_partial_assignment():
    n = Normal()
    m = n.with_params(scale=3)
    assert m.loc() == 0
    assert m.scale() == 3


def test_simple_fit(dataset):
    std_fit = Normal().fit(dataset)
    kernel_fit = Normal(loc=kernels.constant()).fit(dataset)
    assert std_fit.loc() == approx(kernel_fit.loc())


def test_fit_fixed_param(dataset):
    n = Normal().fit(dataset, loc=5)
    assert n.loc() == 5


def test_fit_fixed_param_depth_2(dataset, linear_kernel):
    n = Normal(loc=linear_kernel)
    m = n.fit(dataset, loc_a=5)
    assert m.loc.a() == 5


def test_fit_fixed_param_depth_3(dataset):
    covariate = np.arange(len(dataset))
    n = Normal(loc=kernels.linear(covariate, a=kernels.linear(covariate)))
    m = n.fit(dataset, loc_a_a=5)
    assert m.loc.a.a() == 5


def test_rvs():
    n = Normal(loc=1)
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
    n = Normal(loc=1)
    data = n.rvs(10000)
    trunc_data = data[data >= 0]
    truncated = TruncatedDistribution(Normal(), lower_bound=0)
    fitted_all_data = truncated.fit(data)
    fitted_trunc = truncated.fit(trunc_data)
    for p_trunc, p_all in zip(
        fitted_trunc.flattened_params, fitted_all_data.flattened_params
    ):
        assert p_trunc() == p_all()


def test_distribution_fit_with_shared_params_in_trends():
    """
    when 2 trends in the distribution parameters share a common parameter, e.g. alpha in the below example, the fit results in
    different values for the trends parameters that should be equal.
    """
    x = np.array(np.random.uniform(size=200))
    y = np.array(np.random.normal(size=200))
    alpha = Parameter(0.0)
    n = Normal().fit(y, loc=linear(x=x, b=alpha), scale=linear(x=x, b=alpha))
    alpha1 = n.loc.b
    alpha2 = n.scale.b
    assert alpha1 == alpha2


def test_fit_fixing_shared_params_in_trends():
    """
    when 2 trends in the distribution parameters share a common parameter,
    e.g. alpha in the below example, making one of the corresponding trend parameter
    constant should automatically result in the other trend parameter being constant.
    """
    x = np.array(np.random.uniform(size=200))
    y = np.array(np.random.normal(size=200))
    alpha = Parameter(0.0)
    n = Normal().fit(y, loc=linear(x=x, b=alpha), scale=linear(x=x, b=alpha))
    fixed_alpha = ConstantParameter(
        n.loc.b.value
    )  # should be equal to fit.scale.b as per previous test above
    fit_with_fixed_alpha = n.fit(data=y, loc_b=fixed_alpha)
    assert isinstance(fit_with_fixed_alpha.scale.b, ConstantParameter)
    assert fit_with_fixed_alpha.scale.b.value == fixed_alpha.value


def test_fitted_distribution():
    n = Normal()
    data = n.rvs(1000)
    fitted = n.fit(data)
    assert fitted.loc.value == approx(0.0)
    assert fitted.scale.value == approx(1.0)


def test_fitted_distribution_confidence_interval():
    n = Normal()
    data = n.rvs(1000)
    fitted = n.fit(data)
    ci = fitted.confidence_interval("loc")
    assert len(ci) == 2
    assert ci[0] < ci[1]
    assert ci[0] <= fitted.loc.value <= ci[1]


def test_scipy_distributions_coverage():
    missing = []
    for name, obj in vars(stats).items():
        if isinstance(obj, stats.rv_continuous):
            pykelihood_name = _name_from_scipy_dist(obj)
            pykelihood_dist = getattr(distributions, pykelihood_name, None)
            if pykelihood_dist is None:
                missing.append(name)
    if missing:
        raise AssertionError(
            f"Missing pykelihood distributions for the following scipy distributions: {', '.join(missing)}"
        )


def test_distributions_naming_from_scipy():
    special_cases = {
        # this one is our own alias
        "Normal": "Norm",
        # in these cases the SciPy class names do not match the name of the distribution
        "Trapz": "Trapezoid",
        "Loguniform": "Reciprocal",
        "VonmisesLine": "Vonmises",
    }
    issues = []
    for defined_name, dist_class in vars(distributions).items():
        if isinstance(dist_class, type) and issubclass(
            dist_class, distributions.Distribution
        ):
            alias = special_cases.get(defined_name, defined_name)
            if alias != dist_class.__name__:
                issues.append((alias, dist_class.__name__))
    assert not issues, (
        f"Distribution class names do not match their defined names: {issues}"
    )
