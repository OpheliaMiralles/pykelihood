import numpy as np
import pytest
from scipy import stats

from pykelihood.distributions.scipy_wrappers import Normal
from pykelihood.parameters import Parameter
from pykelihood.parametric import brier_score, crps, quantile_score


def test_crps_is_mean_of_individual_normal_scores_at_explicit_state() -> None:
    location = Parameter(init=0.0)
    model = Normal(loc=location, scale=1.2)
    observations = np.linspace(-2.5, 2.5, 100)
    state = {location: np.asarray(0.5)}
    z = (observations - 0.5) / 1.2
    expected = np.mean(
        1.2
        * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    )

    assert isinstance(crps(model, observations, state=state), float)
    assert crps(model, observations, state=state) == pytest.approx(expected, rel=1e-6)


def test_brier_and_quantile_scores_use_observation_varying_predictions() -> None:
    location = Parameter(init=0.0)
    offsets = np.array([-0.5, 0.0, 0.75])
    model = Normal(loc=location + offsets, scale=1.0)
    observations = np.array([-1.0, 0.5, 2.0])
    state = {location: np.asarray(0.5)}

    exceedance = stats.norm.sf(0.25, loc=0.5 + offsets)
    expected_brier = np.mean((exceedance - (observations >= 0.25)) ** 2)
    expected_median_score = np.mean(np.abs(observations - (0.5 + offsets))) / 2

    assert brier_score(model, observations, 0.25, state=state) == pytest.approx(
        expected_brier
    )
    assert quantile_score(model, observations, 0.5, state=state) == pytest.approx(
        expected_median_score
    )


def test_predictive_scores_reject_misaligned_model_batch() -> None:
    model = Normal(loc=np.array([0.0, 1.0]), scale=1.0)
    observations = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Expected a scalar prediction"):
        brier_score(model, observations, 0.0)


def test_brier_rejects_nonfinite_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be finite"):
        brier_score(Normal(loc=0.0, scale=1.0), [0.0, 1.0], np.nan)
