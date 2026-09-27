import numpy as np
import pytest
from scipy import stats

from pykelihood import probability as p


def test_public_api_fits_reparameterized_exponential_rate() -> None:
    rate = p.Parameter(init=1.0, transform=p.PositiveTransform(), name="rate")
    model = p.Expon(loc=0.0, scale=1 / rate)
    data = np.array([0.3, 0.7, 1.2, 1.8, 2.1])

    fit = p.fit_mle(model, data)

    assert fit.state[rate] == pytest.approx(1 / data.mean(), rel=1e-3)
    assert p.log_likelihood(model, data, state=fit.state) == pytest.approx(
        np.sum(stats.expon.logpdf(data, scale=1 / fit.state[rate]))
    )
    assert isinstance(p.aic(fit, data), float)


def test_public_beta_and_pareto_use_native_scipy_parameters() -> None:
    beta = p.Beta(a=2.0, b=3.0, loc=0.0, scale=1.0)
    pareto = p.Pareto(b=2.0, loc=0.0, scale=1.0)

    assert beta.log_prob(0.4) == pytest.approx(stats.beta.logpdf(0.4, a=2, b=3))
    assert pareto.log_prob(1.5) == pytest.approx(stats.pareto.logpdf(1.5, b=2))
    assert p.fit_mle(beta, [0.2, 0.5]).state == {}
