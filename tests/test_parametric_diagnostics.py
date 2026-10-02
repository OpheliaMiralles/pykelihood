import numpy as np
import pytest
from scipy import stats

from pykelihood.distributions.scipy_wrappers import Normal
from pykelihood.likelihood import log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.parametric import aic, bic, fit_mle


def test_information_criteria_use_fitted_state_and_only_free_coordinates() -> None:
    observations = np.array([-1.0, 0.5, 1.5, 2.0])
    location = Parameter(init=0.0)
    scale = Parameter(init=1.0)
    model = Normal(loc=location, scale=scale)
    fit = fit_mle(model, observations, fixed={location: 0.5})

    expected_log_likelihood = float(
        np.sum(stats.norm.logpdf(observations, loc=0.5, scale=fit.state[scale]))
    )
    assert isinstance(aic(fit, observations), float)
    assert isinstance(bic(fit, observations), float)
    assert aic(fit, observations) == pytest.approx(2 - 2 * expected_log_likelihood)
    assert bic(fit, observations) == pytest.approx(
        np.log(len(observations)) - 2 * expected_log_likelihood
    )


def test_likelihood_rejects_extra_model_batch_axis() -> None:
    model = Normal(loc=np.array([0.0, 1.0]), scale=1.0)
    observations = np.array([[0.0], [1.0], [2.0]])

    with pytest.raises(ValueError, match="Expected log_prob shape"):
        log_likelihood(model, observations)


@pytest.mark.parametrize("criterion", [aic, bic])
def test_information_criteria_reject_failed_fits(criterion) -> None:
    observations = np.array([0.0, 1.0, 2.0])
    fit = fit_mle(Normal(), observations, scipy_args={"options": {"maxiter": 0}})
    assert not fit.optimize_result.success

    with pytest.raises(ValueError, match="successful finite fit"):
        criterion(fit, observations)


@pytest.mark.parametrize("criterion", [aic, bic])
@pytest.mark.parametrize("observation", [np.nan, np.inf])
def test_information_criteria_reject_nonfinite_evaluated_likelihood(
    criterion, observation
) -> None:
    fit = fit_mle(Normal(loc=0.0, scale=1.0), [0.0, 1.0])

    with pytest.raises(ValueError, match="finite likelihood"):
        criterion(fit, [observation])
