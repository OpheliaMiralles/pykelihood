from __future__ import annotations

import numpy as np
from scipy import stats as _stats

from pykelihood.distributions.base import Distribution
from pykelihood.distributions.base import ScipyDistribution
from pykelihood.generic_types import Obs
from pykelihood.utils import ifnone

__all__ = [
    "Exponential",
    "Gamma",
    "Pareto",
    "Beta",
    "GEV",
    "GPD",
    "TruncatedDistribution",
]


class Exponential(ScipyDistribution):
    """
    Exponential distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    rate : float, optional
        Rate parameter, by default 1.0.
    """

    params_names = ("loc", "rate")
    _base_module = _stats.expon

    def __init__(self, loc=0.0, rate=1.0):
        super().__init__(loc, rate)

    def _to_scipy_args(self, loc=None, rate=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        rate : float, optional
            Rate parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        if rate is not None:
            rate = 1 / rate
        return {"loc": ifnone(loc, self.loc()), "scale": ifnone(rate, 1 / self.rate())}


class Gamma(ScipyDistribution):
    """
    Gamma distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    shape : float, optional
        Shape parameter, by default 0.0.
    """

    params_names = ("loc", "scale", "shape")
    _base_module = _stats.gamma

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super().__init__(loc, scale, shape)

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        shape : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "a": ifnone(shape, self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class Pareto(ScipyDistribution):
    """
    Pareto distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    alpha : float, optional
        Shape parameter, by default 1.0.
    """

    params_names = ("loc", "scale", "alpha")
    _base_module = _stats.pareto

    def __init__(self, loc=0.0, scale=1.0, alpha=1.0):
        super().__init__(loc, scale, alpha)

    def _to_scipy_args(self, loc=None, scale=None, alpha=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        alpha : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "c": ifnone(alpha, self.alpha()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class Beta(ScipyDistribution):
    """
    Beta distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    alpha : float, optional
        Alpha parameter, by default 2.0.
    beta : float, optional
        Beta parameter, by default 1.0.
    """

    params_names = ("loc", "scale", "alpha", "beta")
    _base_module = _stats.beta

    def __init__(self, loc=0.0, scale=1.0, alpha=2.0, beta=1.0):
        super().__init__(loc, scale, alpha, beta)

    def _to_scipy_args(self, loc=None, scale=None, alpha=None, beta=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        alpha : float, optional
            Alpha parameter, by default None.
        beta : float, optional
            Beta parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "a": ifnone(alpha, self.alpha()),
            "b": ifnone(beta, self.beta()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class GEV(ScipyDistribution):
    """
    Generalized Extreme Value (GEV) distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    shape : float, optional
        Shape parameter, by default 0.0.

    Notes
    -----
    This version of the Generalized Extreme Value distribution (GEV) does not
    have parameters `c`, `loc`, `scale` but `loc`, `scale` and `shape` where shape
    is `-c`.
    """

    params_names = ("loc", "scale", "shape")
    _base_module = _stats.genextreme

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super().__init__(loc, scale, shape)

    def lb_shape(self, data):
        """
        Calculate the lower bound of the shape parameter.

        Parameters
        ----------
        data : array-like
            Data to calculate the lower bound.

        Returns
        -------
        float
            Lower bound of the shape parameter.
        """
        x_min = data.min()
        x_max = data.max()
        if x_min * x_max < 0:
            return -np.inf
        elif x_min > 0:
            return self.scale / (x_max - self.loc())
        else:
            return self.scale / (x_min - self.loc())

    def ub_shape(self, data):
        """
        Calculate the upper bound of the shape parameter.

        Parameters
        ----------
        data : array-like
            Data to calculate the upper bound.

        Returns
        -------
        float
            Upper bound of the shape parameter.
        """
        x_min = data.min()
        x_max = data.max()
        if x_min * x_max < 0:
            return np.inf
        elif x_min > 0:
            return self.scale / (x_min - self.loc())
        else:
            return self.scale / (x_max - self.loc())

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        shape : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        if shape is not None:
            shape = -shape
        return {
            "c": ifnone(shape, -self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class GPD(ScipyDistribution):
    """
    Generalized Pareto Distribution (GPD).

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    shape : float, optional
        Shape parameter, by default 0.0.
    """

    params_names = ("loc", "scale", "shape")
    _base_module = _stats.genpareto

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super().__init__(loc, scale, shape)

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        shape : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "c": ifnone(shape, self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class TruncatedDistribution(Distribution):
    """
    Truncated distribution.

    Parameters
    ----------
    distribution : Distribution
        The base distribution to truncate.
    lower_bound : float, optional
        Lower bound of the distribution, by default -np.inf.
    upper_bound : float, optional
        Upper bound of the distribution, by default np.inf.

    Raises
    ------
    ValueError
        If the lower and upper bounds are equal.
    """

    params_names = ("distribution",)

    def __init__(
        self, distribution: Distribution, lower_bound=-np.inf, upper_bound=np.inf
    ):
        if upper_bound == lower_bound:
            raise ValueError("Both bounds are equal.")
        super().__init__(distribution)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        lower_cdf = self.distribution.cdf(self.lower_bound)
        upper_cdf = self.distribution.cdf(self.upper_bound)
        self._normalizer = upper_cdf - lower_cdf

    def _build_instance(self, **new_params):
        """
        Build a new instance with the given parameters.

        Parameters
        ----------
        new_params : dict
            New parameters for the instance.

        Returns
        -------
        TruncatedDistribution
            The new instance.
        """
        distribution = new_params.pop("distribution")
        if new_params:
            raise ValueError(f"Unexpected arguments: {new_params}")
        return type(self)(distribution, self.lower_bound, self.upper_bound)

    def _valid_indices(self, x: np.ndarray):
        """
        Get valid indices within the bounds.

        Parameters
        ----------
        x : np.ndarray
            Data to check.

        Returns
        -------
        np.ndarray
            Boolean array of valid indices.
        """
        return (self.lower_bound <= x) & (x <= self.upper_bound)

    def _apply_constraints(self, x):
        """
        Apply constraints to the data.

        Parameters
        ----------
        x : array-like
            Data to apply constraints to.

        Returns
        -------
        array-like
            Data within the bounds.
        """
        return x[self._valid_indices(x)]

    def fit_instance(self, *args, **kwargs):
        """
        Fit the instance to the data.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        TruncatedDistribution
            The fitted instance.
        """
        kwargs.update(lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return super().fit_instance(*args, **kwargs)

    def rvs(self, size: int, *args, **kwargs):
        """
        Generate random variates.

        Parameters
        ----------
        size : int
            Number of random variates to generate.
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        np.ndarray
            Random variates.
        """
        u = Uniform(
            self.distribution.cdf(self.lower_bound),
            self.distribution.cdf(self.upper_bound),
        )
        return self.distribution.inverse_cdf(u.rvs(size, *args, **kwargs))

    def pdf(self, x: Obs):
        """
        Probability density function.

        Parameters
        ----------
        x : Obs
            Data to evaluate.

        Returns
        -------
        np.ndarray
            Probability density values.
        """
        return np.where(
            self._valid_indices(x),
            self.distribution.pdf(x) / self._normalizer,
            0.0,
        )

    def cdf(self, x: Obs):
        """
        Cumulative distribution function.

        Parameters
        ----------
        x : Obs
            Data to evaluate.

        Returns
        -------
        np.ndarray
            Cumulative distribution values.
        """
        right_range_x = (
            self.distribution.cdf(x) - self.distribution.cdf(self.lower_bound)
        ) / self._normalizer
        return np.where(self._valid_indices(x), right_range_x, 0.0)

    def isf(self, q: Obs):
        """
        Inverse survival function.

        Parameters
        ----------
        q : Obs
            Quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Inverse survival function values.
        """
        return self.distribution.isf(
            self.distribution.isf(self.upper_bound) + q * self._normalizer
        )

    def ppf(self, q: Obs):
        """
        Percent point function (inverse of cdf).

        Parameters
        ----------
        q : Obs
            Quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Percent point function values.
        """
        return self.distribution.ppf(
            self.distribution.cdf(self.lower_bound) + q * self._normalizer
        )
