from __future__ import annotations

import typing
from typing import Sequence

import numpy as np

from pykelihood.generic_types import Obs

if typing.TYPE_CHECKING:
    from pykelihood.distributions import Distribution


def likelihood(distribution: Distribution, data: Obs):
    return np.prod(distribution.pdf(data))


def log_likelihood(distribution: Distribution, data: Obs):
    return np.sum(distribution.logpdf(data))


def opposite_log_likelihood(distribution: Distribution, data: Obs):
    return -log_likelihood(distribution, data)


def AIC(distribution: Distribution, data: Obs):
    """
    Akaike Information Criterion: balances the improvement in accuracy of a model with the number of added parameter.
    :return: AIC score: the lower, the better.
    """
    return 2 * opposite_log_likelihood(distribution, data) + 2 * len(
        distribution.optimisation_params
    )


def BIC(distribution: Distribution, data: Obs):
    """
    Bayesian Information Criterion: more info on https://en.wikipedia.org/wiki/Bayesian_information_criterion
    :return: BIC score: the lower, the better.
    """
    return len(distribution.optimisation_params) * np.log(
        len(data)
    ) + 2 * opposite_log_likelihood(distribution, data)


def crps(distribution: Distribution, data: Obs):
    """
    Continuous Rank Probability Score: evaluates the continuous proximity of the empirical cumulative distribution function and that of the
    forecast distribution F.
    :return: \int_{-\infty}^{\infty}(F(y)-H(t-y))^2dy with H the Heavyside function equal to 0. for t<y, 1/2 for t=y and 1 for t>y.
    """
    from scipy import integrate

    def heavyside(t, y):
        return np.where(t < y, 0.0, np.where(t == y, 0.5, 1.0))

    integral = integrate.quad(
        lambda t: (distribution.cdf(t) - np.mean(heavyside(t, data))) ** 2,
        a=-np.inf,
        b=np.inf,
    )
    return integral[0]


def Brier_score(distribution: Distribution, data: Obs, threshold: float = None):
    """
    Brier Score: mean squared error between binary forecast and its empirical value.
    :param threshold: the tail we are interested in predicting correctly.
    :return: mean of (P(Y>=u)-1_{Y>=u})^2
    """
    if threshold is None:
        raise ValueError("This metric requires a input threshold.")
    p_threshold = distribution.sf(threshold)
    return np.mean((p_threshold - (data >= threshold).astype(float)) ** 2)


def quantile_score(distribution: Distribution, data: Obs, quantile: float = None):
    """
    Quantile score: probability weighted score evaluating the difference between the predicted quantile and the
    empirical one.
    :param quantile: quantile of interest.
    :return: q*(y-F^{-1}(q)) if y>=F^{-1}(q), (1-q)*(F^{-1}(q)-y) otherwise
    """
    if quantile is None:
        raise ValueError("This metric requires a input quantile.")
    elif (quantile < 0) or (quantile > 1):
        raise ValueError("The quantile should be between 0 and 1.")

    def rho(x):
        return np.where(x >= 0, x * quantile, x * (quantile - 1))

    return np.mean(rho(data - distribution.inverse_cdf(quantile)))


def qq_l1_distance(distribution: Distribution, data: Obs):
    """
    QQ-Plot-like metrics: mean L1 distance between the x=y line and the (theoretical quantiles, empirical quantiles) one.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_quantile = np.quantile(data, levels)
    return np.mean(np.abs(distribution.inverse_cdf(levels) - empirical_quantile))


def qq_l2_distance(distribution: Distribution, data: Obs):
    """
    QQ-Plot-like metrics: mean L2 distance between the x=y line and the (theoretical quantiles, empirical quantiles) one.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_quantile = np.quantile(data, levels)
    return np.mean((distribution.inverse_cdf(levels) - empirical_quantile) ** 2)


def pp_l1_distance(distribution: Distribution, data: Obs):
    """
    PP-Plot-like metrics: mean L1 distance between the x=y line and the (theoretical cdf, empirical cdf) one.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_cdf = np.quantile(distribution.cdf(data), levels)
    mult = 1 / (np.sqrt(levels * (1 - levels) / np.sqrt(len(data))))
    return np.mean(mult * np.abs(levels - empirical_cdf))


def pp_l2_distance(distribution: Distribution, data: Obs):
    """
    PP-Plot-like metrics: mean L2 distance between the x=y line and the (theoretical cdf, empirical cdf) one.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_cdf = np.quantile(distribution.cdf(data), levels)
    mult = 1 / (np.sqrt(levels * (1 - levels) / np.sqrt(len(data))))
    return np.mean(mult * (levels - empirical_cdf) ** 2)


class ConditioningMethod(object):
    @staticmethod
    def no_conditioning(distribution: Distribution, data: Obs):
        return opposite_log_likelihood(distribution, data)

    @staticmethod
    def excluding_last_obs_rule(distribution: Distribution, data: Obs):
        return ConditioningMethod.no_conditioning(
            distribution, data
        ) - distribution.logpdf(data.iloc[-1])

    @staticmethod
    def partial_conditioning_rule_stopped_obs(
        distribution: Distribution, data: Obs, threshold: Sequence = None
    ):
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        return ConditioningMethod.no_conditioning(
            distribution, data
        ) - distribution.logsf(threshold[-1])

    @staticmethod
    def full_conditioning_rule_stopped_obs(
        distribution: Distribution, data: Obs, threshold: Sequence = None
    ):
        if threshold is None:
            raise ValueError("This metric requires a input threshold.")
        return (
            ConditioningMethod.no_conditioning(distribution, data)
            - distribution.logsf(threshold[-1])
            + np.sum(distribution.logcdf(threshold[:-1]))
        )
