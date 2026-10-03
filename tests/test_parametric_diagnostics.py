from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from pykelihood.distributions.core import ParameterState
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
    observations[:] = 100.0
    assert isinstance(aic(fit), float)
    assert isinstance(bic(fit), float)
    assert aic(fit) == pytest.approx(2 - 2 * expected_log_likelihood)
    assert bic(fit) == pytest.approx(
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
        criterion(fit)


@pytest.mark.parametrize("criterion", [aic, bic])
@pytest.mark.parametrize("location_value", [np.nan, np.inf])
def test_information_criteria_reject_nonfinite_evaluated_likelihood(
    criterion, location_value
) -> None:
    location = Parameter(init=0.0)
    fit = fit_mle(Normal(loc=location, scale=1.0), [0.0, 1.0])
    fit.state[location] = np.asarray(location_value)

    with pytest.raises(ValueError, match="finite likelihood"):
        criterion(fit)


def test_bic_counts_scalar_data_as_one_observation() -> None:
    location = Parameter(init=0.0)
    fit = fit_mle(Normal(loc=location, scale=1.0), 1.5)

    assert fit.data.shape == ()
    assert bic(fit) == pytest.approx(-2 * stats.norm.logpdf(1.5, loc=1.5))


def test_bic_counts_joint_observations_not_event_coordinates() -> None:
    class IndependentNormals(Normal):
        def log_prob(
            self, x: npt.ArrayLike, *, state: ParameterState | None = None
        ) -> npt.NDArray[np.float64]:
            return np.sum(super().log_prob(x, state=state), axis=-1)

    observations = np.array([[-1.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    location = Parameter(init=0.0)
    fit = fit_mle(IndependentNormals(loc=location, scale=1.0), observations)
    expected_log_likelihood = np.sum(
        stats.multivariate_normal.logpdf(observations, mean=np.full(2, 1.5))
    )

    assert bic(fit) == pytest.approx(np.log(3) - 2 * expected_log_likelihood)


def test_bic_rejects_empty_fitting_data() -> None:
    fit = fit_mle(Normal(loc=0.0, scale=1.0), [])

    with pytest.raises(ValueError, match="at least one observation"):
        bic(fit)
