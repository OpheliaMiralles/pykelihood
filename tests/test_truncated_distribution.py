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
    assert_allclose(model.pdf(values), stats.truncnorm.pdf(values, a=0.0, b=2.0))
    assert_allclose(model.logpdf(values), stats.truncnorm.logpdf(values, a=0.0, b=2.0))
    assert_allclose(model.cdf(values), stats.truncnorm.cdf(values, a=0.0, b=2.0))


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

    state = {location: np.asarray(1.0)}
    values = np.asarray([1.0, 1.5, 2.0])
    expected = stats.truncnorm.pdf(values, a=0.0, b=1.0, loc=1.0)
    assert_allclose(model.pdf(values, state=state), expected)


def test_ppf_and_seeded_sampling_stay_inside_truncation_interval() -> None:
    model = TruncatedDistribution(Normal(loc=0.0, scale=1.0), -1.0, 2.0)
    q = np.array([0.0, 0.25, 0.75, 1.0])
    quantiles = model.ppf(q)
    assert_allclose(model.cdf(quantiles), q)
    assert_allclose(quantiles[[0, -1]], [-1.0, 2.0])
    assert np.isnan(model.ppf([-0.1, 1.1])).all()
    first = model.rvs(100, random_state=np.random.default_rng(12))
    second = model.rvs(100, random_state=np.random.default_rng(12))
    assert_allclose(first, second)
    assert np.all((first >= -1.0) & (first <= 2.0))


def test_invalid_intervals_and_out_of_bounds_data_are_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="upper_bound must be greater"):
        TruncatedDistribution(Normal(), lower_bound=2.0, upper_bound=1.0).pdf(1.5)

    with pytest.raises(ValueError, match="no finite likelihood"):
        fit_mle(TruncatedDistribution(Normal(loc=0.0, scale=1.0), 2.0, 1.0), [1.5])

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


def test_fit_mle_can_reject_invalid_free_bound_proposals() -> None:
    lower = Parameter(init=0.0)
    upper = Parameter(init=1.0)
    model = TruncatedDistribution(Normal(loc=0.0, scale=1.0), lower, upper)
    initial_simplex = np.array([[0.0, 1.0], [2.0, 1.0], [0.0, 0.9]])

    result = fit_mle(
        model,
        [0.3, 0.5, 0.7],
        scipy_args={"options": {"initial_simplex": initial_simplex}},
    )

    assert result.optimize_result.success
    assert result.state[lower] <= 0.3
    assert result.state[upper] >= 0.7
