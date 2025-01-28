from __future__ import annotations

import typing
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.special import binom

from pykelihood.generic_types import Obs

if typing.TYPE_CHECKING:
    from pykelihood.distributions import Distribution


def bootstrap(
    metric: Callable[[Distribution, Obs], float],
    bootstrap_method: Callable[[Obs], Iterable[Obs]],
):
    """
    Bootstrap utility.

    Parameters
    ----------
    metric : Callable[[Distribution, Obs], float]
        ``pykelihood.metrics`` object.
    bootstrap_method : Callable[[Obs], Iterable[Obs]]
        Function that can be called on ``Obs`` or sequences of ``Obs``. For example, ``np.random.sample``.

    Returns
    -------
    Callable
        Value of the metric of interest averaged over bootstrapped data.
    """

    def bootstrapped(distribution, data):
        datasets = bootstrap_method(data)
        sizes = [len(d) for d in datasets]
        return np.sum([len(d) * metric(distribution, d) for d in datasets]) / np.sum(
            sizes
        )

    return bootstrapped


def likelihood(distribution: Distribution, data: Obs):
    """
    Standard likelihood function.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``, e.g. ``Union[float, Sequence[float]]``.

    Returns
    -------
    float
        Likelihood value.
    """
    return np.prod(distribution.pdf(data))


def log_likelihood(distribution: Distribution, data: Obs):
    """
    Log-likelihood function.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        Log-likelihood value.
    """
    return np.sum(distribution.logpdf(data))


def opposite_log_likelihood(distribution: Distribution, data: Obs):
    """
    Opposite log-likelihood function.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        Opposite log-likelihood value.
    """
    return -log_likelihood(distribution, data)


def AIC(distribution: Distribution, data: Obs):
    """
    Akaike Information Criterion (AIC).

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        AIC score: the lower, the better.
    """
    return 2 * opposite_log_likelihood(distribution, data) + 2 * len(
        distribution.optimisation_params
    )


def BIC(distribution: Distribution, data: Obs):
    """
    Bayesian Information Criterion (BIC).

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        BIC score: the lower, the better.
    """
    return len(distribution.optimisation_params) * np.log(
        len(data)
    ) + 2 * opposite_log_likelihood(distribution, data)


def crps(distribution: Distribution, data: Obs):
    """
    Continuous Rank Probability Score (CRPS).

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        CRPS value.

    Notes
    -----
    Evaluates the continuous proximity of the empirical cumulative distribution function and that of the forecast distribution F.
    \int_{-\infty}^{\infty}(F(y)-H(t-y))^2dy with H the Heavyside function equal to 0. for t<y, 1/2 for t=y and 1 for t>y.
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
    Brier Score.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    threshold : float, optional
        The tail we are interested in predicting correctly.

    Returns
    -------
    float
        Mean of (P(Y>=u)-1_{Y>=u})^2.

    Raises
    ------
    ValueError
        If threshold is None.

    Notes
    -----
    Mean squared error between binary forecast and its empirical value.
    """
    if threshold is None:
        raise ValueError("This metric requires a input threshold.")
    p_threshold = distribution.sf(threshold)
    return np.mean((p_threshold - (data >= threshold).astype(float)) ** 2)


def quantile_score(distribution: Distribution, data: Obs, quantile: float = None):
    """
    Quantile score.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    quantile : float, optional
        Quantile of interest.

    Returns
    -------
    float
        Quantile score value.

    Raises
    ------
    ValueError
        If quantile is None or not between 0 and 1.

    Notes
    -----
    Probability weighted score evaluating the difference between the predicted quantile and the empirical one.
    q*(y-F^{-1}(q)) if y>=F^{-1}(q), (1-q)*(F^{-1}(q)-y) otherwise.
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
    QQ-Plot-like metrics: mean L1 distance.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        Mean L1 distance.

    Notes
    -----
    Mean L1 distance between the x=y line and the (theoretical quantiles, empirical quantiles) one.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_quantile = np.quantile(data, levels)
    return np.mean(np.abs(distribution.inverse_cdf(levels) - empirical_quantile))


def qq_l2_distance(distribution: Distribution, data: Obs):
    """
    QQ-Plot-like metrics: mean L2 distance.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        Mean L2 distance.

    Notes
    -----
    Mean L2 distance between the x=y line and the (theoretical quantiles, empirical quantiles) one.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_quantile = np.quantile(data, levels)
    return np.mean((distribution.inverse_cdf(levels) - empirical_quantile) ** 2)


def pp_l1_distance(distribution: Distribution, data: Obs):
    """
    PP-Plot-like metrics: mean L1 distance.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        Mean L1 distance.

    Notes
    -----
    Mean L1 distance between the x=y line and the (theoretical cdf, empirical cdf) one.
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
    PP-Plot-like metrics: mean L2 distance.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.

    Returns
    -------
    float
        Mean L2 distance.

    Notes
    -----
    Mean L2 distance between the x=y line and the (theoretical cdf, empirical cdf) one.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_cdf = np.quantile(distribution.cdf(data), levels)
    mult = 1 / (np.sqrt(levels * (1 - levels) / np.sqrt(len(data))))
    return np.mean(mult * (levels - empirical_cdf) ** 2)
