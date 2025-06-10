from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Callable

import numpy as np

from pykelihood.generic_types import Obs

if typing.TYPE_CHECKING:
    from pykelihood.distributions import Distribution


def bootstrap(
    metric: Callable[[Distribution, Obs], float],
    bootstrap_method: Callable[[Obs], Iterable[Obs]],
):
    """
    Bootstrap utility. Flexible enough to be used with any metric and bootstrap method.

    Parameters
    ----------
    metric : Callable[[Distribution, Obs], float]
        ``pykelihood.metrics`` object.
    bootstrap_method : Callable[[Obs], Iterable[Obs]]
        Function that can be called on ``Obs`` or sequences of ``Obs``.
        For example, ``np.random.sample``.

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
    r"""
    Continuous Rank Probability Score (CRPS).

    Evaluates the continuous proximity of the empirical cumulative distribution function
    and that of the fitted distribution :math:`F`.

    .. math::

        CRPS = \int_{-\infty}^{\infty}(F(t)-\frac{1}{N}\sum_{i=1}^N H(t,y_i))^2dt,

    where :math:`F` is the cumulative distribution function of the fitted distribution and
    with H the Heavyside function equal to 0 for :math:`t<y`, 1/2 for :math:`t=y`
    and 1 for :math:`t>y`.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
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
    r"""
    Brier Score.

    Represents the mean squared error between observed
    exceedance of a threshold :math:`u` and the value of the fitted survival
    function at :math:`u`.

    .. math::

         BS = \frac{1}{N}\sum_{i=1}^{N}(\bar{F}(u)-1_{y_i\geq u})^2,

    where :math:`\bar{F}` is the fitted survival function
    and :math:`1_{y_i\geq u}` is the indicator function that the observed value
    is above threshold :math:`u`.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    threshold : float, optional
        The tail we are interested in predicting correctly.

    Raises
    ------
    ValueError
        If threshold is None.
    """
    if threshold is None:
        raise ValueError("This metric requires a input threshold.")
    p_threshold = distribution.sf(threshold)
    return np.mean((p_threshold - (data >= threshold).astype(float)) ** 2)


def quantile_score(distribution: Distribution, data: Obs, quantile: float):
    r"""
    Quantile score.

    Probability weighted score evaluating the difference
    between the fitted quantile and the observed one.

    .. math::

        QS = \begin{cases}
         q \cdot (y - F^{-1}(q)) & \text{if } y \geq F^{-1}(q), \\
        (1 - q) \cdot (F^{-1}(q) - y) & \text{otherwise},
        \end{cases}

    where :math:`F^{-1}(q)` is the fitted inverse cumulative distribution
    function at quantile :math:`q`.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    quantile : float
        Quantile of interest.

    Raises
    ------
    ValueError
        If quantile is not between 0 and 1.
    """
    if (quantile < 0) or (quantile > 1):
        raise ValueError("The quantile should be between 0 and 1.")

    def rho(x):
        return np.where(x >= 0, x * quantile, x * (quantile - 1))

    return np.mean(rho(data - distribution.inverse_cdf(quantile)))


def qq_l1_distance(distribution: Distribution, data: Obs):
    """
    QQ-Plot-like metric: mean L1 distance.

    Mean L1 distance between the x=y line and the one defined by
    x=fitted quantiles, y=observed quantiles.

    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_quantile = np.quantile(data, levels)
    return np.mean(np.abs(distribution.inverse_cdf(levels) - empirical_quantile))


def qq_l2_distance(distribution: Distribution, data: Obs):
    """
    QQ-Plot-like metric: mean L2 distance.

    Mean L2 distance between the x=y line and the one defined by
    x=fitted quantiles, y=observed quantiles.

    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_quantile = np.quantile(data, levels)
    return np.mean((distribution.inverse_cdf(levels) - empirical_quantile) ** 2)


def pp_l1_distance(distribution: Distribution, data: Obs):
    """
    PP-Plot-like metric: mean L1 distance.

    Mean L1 distance between the x=y line and the one defined by
    x=fitted cdf, y=observed cdf.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_cdf = np.quantile(distribution.cdf(data), levels)
    mult = 1 / (np.sqrt(levels * (1 - levels) / np.sqrt(len(data))))
    return np.mean(mult * np.abs(levels - empirical_cdf))


def pp_l2_distance(distribution: Distribution, data: Obs):
    """
    PP-Plot-like metric: mean L2 distance.

    Mean L2 distance between the x=y line the one defined by
    x=fitted cdf, y=observed cdf.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process.
    arXiv preprint arXiv:2102.00884.

    Parameters
    ----------
    distribution : Distribution
        ``pykelihood.Distribution`` object.
    data : Obs
        Data of type ``Obs``.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 100)
    empirical_cdf = np.quantile(distribution.cdf(data), levels)
    mult = 1 / (np.sqrt(levels * (1 - levels) / np.sqrt(len(data))))
    return np.mean(mult * (levels - empirical_cdf) ** 2)
