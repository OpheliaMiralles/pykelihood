from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import stats as _stats

from pykelihood.distributions._compat import CompatibilityValue, as_expr, evaluate
from pykelihood.distributions.base import Distribution, ScipyDistribution
from pykelihood.distributions.core import (
    ParameterDefault,
    ParameterInput,
    ParameterState,
    RandomState,
)
from pykelihood.distributions.scipy import Beta, Gamma, Pareto
from pykelihood.expr import Expr, Node
from pykelihood.parameters import Parameter
from pykelihood.state import PositiveTransform, ProbabilityTransform

__all__ = [
    "Exponential",
    "Gamma",
    "Pareto",
    "Beta",
    "GEV",
    "GPD",
    "TruncatedDistribution",
    "Bernoulli",
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

    _base_module = _stats.expon

    def __init__(self, loc: ParameterInput = None, rate: ParameterInput = None) -> None:
        super().__init__(
            self._base_module,
            {"loc": loc, "rate": rate},
            defaults={
                "loc": ParameterDefault(0.0),
                "rate": ParameterDefault(1.0, PositiveTransform()),
            },
            reparametrization=_exponential_parameters,
        )


def _exponential_parameters(
    parameters: Mapping[str, npt.NDArray[np.float64]],
) -> Mapping[str, npt.NDArray[np.float64]]:
    return {"loc": parameters["loc"], "scale": 1.0 / parameters["rate"]}


class _ShapeCompatibilityDistribution(ScipyDistribution):
    """Legacy ``shape`` naming over one native SciPy ``c`` shape parameter."""

    _base_module: _stats.rv_continuous

    def __init__(
        self,
        loc: ParameterInput = None,
        scale: ParameterInput = None,
        shape: ParameterInput = None,
    ) -> None:
        shape_node = (
            Parameter(init=0.0, name="shape") if shape is None else as_expr(shape)
        )
        native_shape = self._to_native_shape(shape_node)
        super().__init__(
            self._base_module,
            {"loc": loc, "scale": scale, "c": native_shape},
            defaults={
                "loc": ParameterDefault(0.0),
                "scale": ParameterDefault(1.0, PositiveTransform()),
            },
        )
        self._public_parameters = MappingProxyType(
            {
                "loc": self._parameters["loc"],
                "scale": self._parameters["scale"],
                "shape": shape_node,
            }
        )

    @staticmethod
    def _to_native_shape(shape: Expr) -> Expr:
        raise NotImplementedError

    @property
    def parameters(self) -> Mapping[str, Expr]:
        return self._public_parameters

    def _with_parameters(
        self, parameters: Mapping[str, Node]
    ) -> _ShapeCompatibilityDistribution:
        try:
            loc = cast(Expr, parameters["loc"])
            scale = cast(Expr, parameters["scale"])
            shape = cast(Expr, parameters["shape"])
        except KeyError as error:
            raise ValueError("Expected loc, scale, and shape parameters.") from error
        return type(self)(loc=loc, scale=scale, shape=shape)

    def _scipy_parameters(
        self, state: ParameterState | None
    ) -> dict[str, npt.NDArray[np.float64]]:
        parameter_state = {} if state is None else state
        public = {
            name: evaluate(node, parameter_state)
            for name, node in self.parameters.items()
        }
        return {
            "c": self._native_shape_value(public["shape"]),
            "loc": public["loc"],
            "scale": public["scale"],
        }

    @staticmethod
    def _native_shape_value(shape: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError


class GEV(_ShapeCompatibilityDistribution):
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

    _base_module = _stats.genextreme

    def __init__(
        self,
        loc: ParameterInput = None,
        scale: ParameterInput = None,
        shape: ParameterInput = None,
    ) -> None:
        super().__init__(loc=loc, scale=scale, shape=shape)

    @staticmethod
    def _to_native_shape(shape: Expr) -> Expr:
        return -shape

    @staticmethod
    def _native_shape_value(shape: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return -shape

    def lb_shape(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
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
        values = self._scipy_parameters(None)
        x_min, x_max = np.min(data), np.max(data)
        if x_min * x_max < 0:
            return np.asarray(-np.inf)
        if x_min > 0:
            return values["scale"] / (x_max - values["loc"])
        return values["scale"] / (x_min - values["loc"])

    def ub_shape(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
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
        values = self._scipy_parameters(None)
        x_min, x_max = np.min(data), np.max(data)
        if x_min * x_max < 0:
            return np.asarray(np.inf)
        if x_min > 0:
            return values["scale"] / (x_min - values["loc"])
        return values["scale"] / (x_max - values["loc"])


class GPD(_ShapeCompatibilityDistribution):
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

    _base_module = _stats.genpareto

    def __init__(
        self,
        loc: ParameterInput = None,
        scale: ParameterInput = None,
        shape: ParameterInput = None,
    ) -> None:
        super().__init__(loc=loc, scale=scale, shape=shape)

    @staticmethod
    def _to_native_shape(shape: Expr) -> Expr:
        return shape

    @staticmethod
    def _native_shape_value(shape: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return shape


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

    def __init__(
        self, distribution: Distribution, lower_bound=-np.inf, upper_bound=np.inf
    ) -> None:
        if upper_bound == lower_bound:
            raise ValueError("Both bounds are equal.")
        self.distribution = distribution
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._parameters = MappingProxyType({"distribution": cast(Expr, distribution)})

    @property
    def parameters(self) -> Mapping[str, Expr]:
        return self._parameters

    @property
    def params_names(self) -> tuple[str, ...]:
        return ("distribution",)

    def _build_instance(
        self, distribution: Distribution, **new_params: object
    ) -> TruncatedDistribution:
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
        if new_params:
            raise ValueError(f"Unexpected arguments: {new_params}")
        return type(self)(distribution, self.lower_bound, self.upper_bound)

    def _with_parameters(self, parameters: Mapping[str, Node]) -> TruncatedDistribution:
        distribution = parameters.get("distribution")
        if not isinstance(distribution, Distribution):
            raise TypeError("TruncatedDistribution requires a Distribution child.")
        return self._build_instance(distribution)

    def __getattr__(self, name: str) -> Parameter | CompatibilityValue:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return cast(
                Parameter | CompatibilityValue, getattr(self.distribution, name)
            )

    def _valid_indices(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
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

    def _apply_constraints(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
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
        values = np.asarray(data, dtype=np.float64)
        return values[self._valid_indices(values)]

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.float64]:
        """
        Generate random variates.

        Parameters
        ----------
        size : int
            Number of random variates to generate.

        Returns
        -------
        np.ndarray
            Random variates.
        """
        lower = self.distribution.cdf(self.lower_bound, state=state)
        upper = self.distribution.cdf(self.upper_bound, state=state)
        quantiles = _stats.uniform.rvs(
            loc=lower, scale=upper - lower, size=size, random_state=random_state
        )
        return self.distribution.ppf(quantiles, state=state)

    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Probability density function.

        Parameters
        ----------
        x : array-like
            Data to evaluate.

        Returns
        -------
        np.ndarray
            Probability density values.
        """
        values = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = self.distribution.pdf(values, state=state) / self._normalizer(
                state
            )
        return np.asarray(
            np.where(self._valid_indices(values), result, 0.0), dtype=np.float64
        )

    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        values = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = self.distribution.logpdf(values, state=state) - np.log(
                self._normalizer(state)
            )
        return np.asarray(
            np.where(self._valid_indices(values), result, -np.inf), dtype=np.float64
        )

    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Cumulative distribution function.

        Parameters
        ----------
        x : array-like
            Data to evaluate.

        Returns
        -------
        np.ndarray
            Cumulative distribution values.
        """
        values = np.asarray(x, dtype=np.float64)
        lower = self.distribution.cdf(self.lower_bound, state=state)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (
                self.distribution.cdf(values, state=state) - lower
            ) / self._normalizer(state)
        return np.asarray(
            np.where(
                values < self.lower_bound,
                0.0,
                np.where(values > self.upper_bound, 1.0, result),
            ),
            dtype=np.float64,
        )

    def isf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Inverse survival function.

        Parameters
        ----------
        q : array-like
            Quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Inverse survival function values.
        """
        return self.ppf(1.0 - np.asarray(q, dtype=np.float64), state=state)

    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Percent point function (inverse of cdf).

        Parameters
        ----------
        q : array-like
            Quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Percent point function values.
        """
        lower = self.distribution.cdf(self.lower_bound, state=state)
        return self.distribution.ppf(
            lower + np.asarray(q, dtype=np.float64) * self._normalizer(state),
            state=state,
        )

    def _normalizer(self, state: ParameterState | None) -> npt.NDArray[np.float64]:
        return self.distribution.cdf(
            self.upper_bound, state=state
        ) - self.distribution.cdf(self.lower_bound, state=state)


class Bernoulli(Distribution):
    """Bernoulli distribution using ``pdf`` as the public mass vocabulary."""

    def __init__(self, p: ParameterInput = None) -> None:
        if p is None:
            probability: Expr = Parameter(
                init=0.5, name="p", transform=ProbabilityTransform()
            )
        else:
            probability = as_expr(p)
        self._parameters = MappingProxyType({"p": probability})

    @property
    def parameters(self) -> Mapping[str, Expr]:
        return self._parameters

    def _probability(self, state: ParameterState | None) -> npt.NDArray[np.float64]:
        return evaluate(self.parameters["p"], {} if state is None else state)

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.rvs(
                self._probability(state), size=size, random_state=random_state
            ),
            dtype=np.float64,
        )

    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.cdf(x, self._probability(state)), dtype=np.float64
        )

    def isf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.isf(q, self._probability(state)), dtype=np.float64
        )

    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.ppf(q, self._probability(state)), dtype=np.float64
        )

    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.pmf(x, self._probability(state)), dtype=np.float64
        )

    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.logpmf(x, self._probability(state)), dtype=np.float64
        )

    def sf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.sf(x, self._probability(state)), dtype=np.float64
        )

    def logcdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.logcdf(x, self._probability(state)), dtype=np.float64
        )

    def logsf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            _stats.bernoulli.logsf(x, self._probability(state)), dtype=np.float64
        )
