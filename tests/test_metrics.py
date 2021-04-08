from functools import partial

import numpy as np
import pytest
from pytest import approx
from scipy.stats import norm

from pykelihood import kernels
from pykelihood.distributions import GEV, GPD, Normal
from pykelihood.metrics import (
    Brier_score,
    crps,
    log_likelihood,
    opposite_log_likelihood,
    pp_l1_distance,
    pp_l2_distance,
    qq_l1_distance,
    qq_l2_distance,
    quantile_score,
)


@pytest.fixture
def make_data():
    def f(dist, shape):
        return dist(shape=shape).rvs(100000)

    return f


@pytest.fixture
def normal_data():
    return np.array(Normal().rvs(100000))


def test_log_likelihood_is_correct(normal_data):
    ndata = normal_data[:1000]
    n = Normal(loc=kernels.linear(np.arange(len(ndata)))).fit_instance(ndata)
    actual = log_likelihood(n, ndata)
    expected = sum(
        norm.logpdf(x, loc=loc, scale=n.scale()) for x, loc in zip(ndata, n.loc())
    )
    assert actual == approx(expected)


@pytest.mark.parametrize("shape", [-0.5, 0.0, 0.5])
def test_gev(shape):
    """
    tests based on observations made by Friederichs, P., & Thorarinsdottir, T. L. (2012).
    Forecast verification for extreme value distributions with an application to probabilistic
    peak wind prediction. Environmetrics, 23(7), 579-594.
    :param shape: varying shape parameter
    :return:
    """
    y = np.linspace(-4, 4, 10)
    gev = GEV(shape=shape)
    # CRPS Score
    crps_gev = np.array([crps(gev, x) for x in y])
    assert np.all(crps_gev >= 0)
    # QS Score
    qs_gev = np.array([quantile_score(gev, x, quantile=0.5) for x in y])
    assert np.all(qs_gev >= 0)
    # Ignorance Score
    is_gev = np.array([opposite_log_likelihood(gev, x) for x in y])
    assert np.all(is_gev >= 0)
    assert np.all(is_gev - crps_gev >= 0)
    assert np.all(crps_gev - qs_gev >= 0)


@pytest.mark.parametrize("shape", [-0.5, 0.0, 0.5])
def test_gpd(shape):
    """
    tests based on observations made by Friederichs, P., & Thorarinsdottir, T. L. (2012).
    Forecast verification for extreme value distributions with an application to probabilistic
    peak wind prediction. Environmetrics, 23(7), 579-594.
    :param shape: varying shape parameter
    :return:
    """
    y = np.linspace(1, 5, 10)
    gpd = GPD(shape=shape)
    # CRPS Score
    crps_gpd = np.array([crps(gpd, x) for x in y])
    assert np.all(crps_gpd >= 0)
    # QS Score
    qs_gpd = np.array([quantile_score(gpd, x, quantile=0.5) for x in y])
    assert np.all(qs_gpd >= 0)
    # Ignorance Score
    is_gpd = np.array([opposite_log_likelihood(gpd, x) for x in y])
    assert np.all(is_gpd >= 0)
    assert np.all(is_gpd - crps_gpd >= 0)
    assert np.all(crps_gpd - qs_gpd >= 0)


@pytest.mark.parametrize("shape", [-0.5, 0, 0.5])
@pytest.mark.parametrize("distribution", [GEV, GPD], ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    "score_func",
    [
        crps,
        partial(Brier_score, threshold=0),
        opposite_log_likelihood,
        qq_l1_distance,
        qq_l2_distance,
        pp_l1_distance,
        pp_l2_distance,
    ],
)
def test_logical(score_func, distribution, shape, normal_data, make_data):
    data = make_data(distribution, shape)
    instance_dist = distribution(shape=shape)
    score_same_data_distribution = score_func(instance_dist, data)
    score_normal_data = score_func(instance_dist, normal_data)
    assert score_same_data_distribution <= score_normal_data


@pytest.mark.parametrize("shape", [-0.5, 0, 0.5], ids=str)
@pytest.mark.parametrize("distribution", [GEV, GPD], ids=str)
def test_logical_quantile_score(distribution, shape, make_data, normal_data):
    data = make_data(distribution, shape)
    instance_dist = distribution(shape=shape)
    # overall better prediction of the distribution quantiles
    scores = []
    norm_scores = []
    for quantile in [0.25, 0.5, 0.75]:
        y = np.quantile(data, quantile)
        y_norm = np.quantile(normal_data, quantile)
        scores.append(quantile_score(instance_dist, y, quantile=quantile))
        norm_scores.append(quantile_score(instance_dist, y_norm, quantile=quantile))
    assert np.sum(scores) <= np.sum(norm_scores)
    # better prediction of distribution upper tail
    scores = []
    norm_scores = []
    for quantile in [0.85, 0.9, 0.95, 0.99]:
        y = np.quantile(data, quantile)
        y_norm = np.quantile(normal_data, quantile)
        scores.append(quantile_score(instance_dist, y, quantile=quantile))
        norm_scores.append(quantile_score(instance_dist, y_norm, quantile=quantile))
    assert np.sum(scores) <= np.sum(norm_scores)
