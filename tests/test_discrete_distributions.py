import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pykelihood.distributions.core import InvalidDistributionState
from pykelihood.distributions.scipy_wrappers import Bernoulli
from pykelihood.parameters import Parameter
from pykelihood.parametric import fit_mle


def test_bernoulli_log_prob_matches_scipy_and_returns_one_score_per_value() -> None:
    probability = Parameter(init=0.4)
    model = Bernoulli(probability)
    values = np.array([0, 1, 1, 0])
    state = {probability: np.asarray(0.7)}

    actual = model.log_prob(values, state=state)

    assert actual.shape == values.shape
    assert_allclose(actual, stats.bernoulli.logpmf(values, 0.7))
    assert_allclose(model.pmf(values, state=state), stats.bernoulli.pmf(values, 0.7))
    assert tuple(model.parameters.values()) == (probability,)


def test_bernoulli_returns_negative_infinity_outside_its_support() -> None:
    model = Bernoulli(0.4)

    assert_allclose(
        model.log_prob([-1, 0, 1, 2]), [-np.inf, np.log(0.6), np.log(0.4), -np.inf]
    )


def test_bernoulli_rejects_invalid_probability_states() -> None:
    probability = Parameter(init=0.5)
    model = Bernoulli(probability)

    with pytest.raises(InvalidDistributionState, match=r"must be in \[0, 1\]"):
        model.log_prob([0, 1], state={probability: np.asarray(1.1)})


def test_bernoulli_sampling_is_integer_and_reproducible() -> None:
    model = Bernoulli([0.2, 0.8])

    actual = model.rvs(5, random_state=4)
    repeated = model.rvs(5, random_state=4)

    assert actual.shape == (5, 2)
    assert actual.dtype.kind in "iu"
    assert_allclose(actual, repeated)
    assert np.all((actual == 0) | (actual == 1))


def test_bernoulli_uses_the_common_mle_fitter() -> None:
    probability = Parameter(init=0.5)
    model = Bernoulli(probability)

    result = fit_mle(model, [1, 0, 1, 1, 0, 1])

    assert result.optimize_result.success
    assert result.state[probability] == pytest.approx(4 / 6, abs=1e-3)
