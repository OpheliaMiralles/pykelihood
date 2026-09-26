import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pykelihood.distributions.scipy_wrappers import Normal
from pykelihood.distributions.truncated import TruncatedDistribution
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.parametric import fit_mle
from pykelihood.state import ParameterLayout


def test_truncated_pdf_logpdf_and_cdf_respect_numeric_bounds() -> None:
    model = TruncatedDistribution(Normal(loc=0.0, scale=1.0), 0.0, 2.0)
    values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    normalizer = stats.norm.cdf(2.0) - stats.norm.cdf(0.0)

    expected_pdf = np.where(
        (values >= 0.0) & (values <= 2.0), stats.norm.pdf(values) / normalizer, 0.0
    )
    expected_logpdf = np.full(values.shape, -np.inf)
    in_bounds = expected_pdf > 0.0
    expected_logpdf[in_bounds] = np.log(expected_pdf[in_bounds])
    expected_cdf = np.where(
        values < 0.0,
        0.0,
        np.where(
            values >= 2.0,
            1.0,
            (stats.norm.cdf(values) - stats.norm.cdf(0.0)) / normalizer,
        ),
    )

    assert_allclose(model.pdf(values), expected_pdf)
    assert_allclose(model.logpdf(values), expected_logpdf)
    assert_allclose(model.cdf(values), expected_cdf)


def test_truncation_normalizer_uses_current_state_and_shared_nodes_traverse_once() -> (
    None
):
    location = Parameter(init=0.0)
    base = Normal(loc=location, scale=1.0)
    model = TruncatedDistribution(
        base, lower_bound=location, upper_bound=location + 1.0
    )
    layout = ParameterLayout.from_expr(model)

    assert layout.parameters == (location,)
    assert layout.parameter_paths[location] == (
        ("distribution", "loc"),
        ("lower_bound",),
        ("upper_bound", "left"),
    )

    state = {location: np.asarray(1.0)}
    values = np.asarray([1.0, 1.5, 2.0])
    normalizer = stats.norm.cdf(2.0, loc=1.0) - stats.norm.cdf(1.0, loc=1.0)
    expected = stats.norm.pdf(values, loc=1.0) / normalizer
    assert_allclose(model.pdf(values, state=state), expected)


def test_ppf_and_seeded_sampling_stay_inside_truncation_interval() -> None:
    model = TruncatedDistribution(Normal(loc=0.0, scale=1.0), -1.0, 2.0)
    q = np.array([0.0, 0.25, 0.75, 1.0])
    lower_cdf = stats.norm.cdf(-1.0)
    upper_cdf = stats.norm.cdf(2.0)
    expected = stats.norm.ppf(lower_cdf + q * (upper_cdf - lower_cdf))

    assert_allclose(model.ppf(q), expected)
    first = model.rvs(100, random_state=np.random.default_rng(12))
    second = model.rvs(100, random_state=np.random.default_rng(12))
    assert_allclose(first, second)
    assert np.all((first >= -1.0) & (first <= 2.0))


def test_invalid_intervals_and_out_of_bounds_data_are_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="upper_bound must be greater"):
        TruncatedDistribution(Normal(), lower_bound=2.0, upper_bound=1.0).pdf(1.5)

    model = TruncatedDistribution(Normal(loc=0.0, scale=1.0), 0.0, 1.0)
    assert np.isinf(negative_log_likelihood(model, [1.5]))


def test_fit_mle_traverses_nested_distribution_and_returns_fitted_state() -> None:
    location = Parameter(init=0.0)
    model = TruncatedDistribution(Normal(loc=location, scale=1.0), 0.0, 1.0)
    data = [0.2, 0.4, 0.7, 0.8]
    initial_score = negative_log_likelihood(model, data)

    result = fit_mle(model, data)

    assert result.model is model
    assert result.optimize_result.success
    assert location in result.state
    assert negative_log_likelihood(model, data, state=result.state) < initial_score
