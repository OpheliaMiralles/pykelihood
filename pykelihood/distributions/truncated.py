"""Opt-in wrappers for distributions with explicit truncation bounds."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Union

import numpy as np
import numpy.typing as npt

from pykelihood.distributions.core import Distribution, ParameterState, RandomState
from pykelihood.expr import Constant, Expr, Node, PathElem

BoundInput = Union[Expr, npt.ArrayLike]


def _as_expr(value: BoundInput) -> Expr:
    return value if isinstance(value, Expr) else Constant(value)


def _rng(random_state: RandomState) -> np.random.Generator | np.random.RandomState:
    if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        return random_state
    return np.random.default_rng(random_state)


class TruncatedDistribution(Distribution):
    """Condition a continuous distribution to lie within ``[lower, upper]``.

    Bounds may be literals or state-evaluable expressions. The wrapped model and
    bounds remain graph children, so fitting discovers shared parameters through
    the ordinary node traversal.
    """

    def __init__(
        self,
        distribution: Distribution,
        lower_bound: BoundInput = -np.inf,
        upper_bound: BoundInput = np.inf,
    ) -> None:
        self.distribution = distribution
        self.lower_bound = _as_expr(lower_bound)
        self.upper_bound = _as_expr(upper_bound)
        self._parameters = MappingProxyType(
            {"lower_bound": self.lower_bound, "upper_bound": self.upper_bound}
        )

    @property
    def parameters(self) -> Mapping[str, Expr]:
        """The truncation bounds; wrapped distribution is a separate child."""
        return self._parameters

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        yield "distribution", self.distribution
        yield from self.parameters.items()

    def _bounds(self, state: ParameterState | None) -> tuple[npt.NDArray, npt.NDArray]:
        parameter_state = {} if state is None else state
        lower = np.asarray(self.lower_bound.eval(parameter_state), dtype=np.float64)
        upper = np.asarray(self.upper_bound.eval(parameter_state), dtype=np.float64)
        if np.any(upper <= lower):
            raise ValueError("upper_bound must be greater than lower_bound.")
        return lower, upper

    def _normalizer(
        self, state: ParameterState | None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        lower, upper = self._bounds(state)
        mass = np.asarray(
            self.distribution.cdf(upper, state=state)
            - self.distribution.cdf(lower, state=state),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("Truncation interval must have positive probability mass.")
        return lower, upper, mass

    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        lower, upper, mass = self._normalizer(state)
        values = np.asarray(x, dtype=np.float64)
        density = self.distribution.pdf(values, state=state) / mass
        return np.asarray(np.where((values >= lower) & (values <= upper), density, 0.0))

    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        lower, upper, mass = self._normalizer(state)
        values = np.asarray(x, dtype=np.float64)
        log_density = self.distribution.logpdf(values, state=state) - np.log(mass)
        return np.asarray(
            np.where((values >= lower) & (values <= upper), log_density, -np.inf)
        )

    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        lower, upper, mass = self._normalizer(state)
        values = np.asarray(x, dtype=np.float64)
        lower_cdf = self.distribution.cdf(lower, state=state)
        conditional = (self.distribution.cdf(values, state=state) - lower_cdf) / mass
        return np.asarray(
            np.where(values < lower, 0.0, np.where(values >= upper, 1.0, conditional))
        )

    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        lower, _upper, mass = self._normalizer(state)
        lower_cdf = self.distribution.cdf(lower, state=state)
        return np.asarray(
            self.distribution.ppf(
                lower_cdf + np.asarray(q, dtype=np.float64) * mass, state=state
            ),
            dtype=np.float64,
        )

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.float64]:
        uniforms = _rng(random_state).uniform(size=size)
        return self.ppf(uniforms, state=state)
