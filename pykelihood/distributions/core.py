"""Explicit-state distribution nodes for the next-generation API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Union

import numpy as np
import numpy.typing as npt

from pykelihood.expr import Expr, Node, PathElem
from pykelihood.parameters import Parameter

ParameterInput = Union[Expr, npt.ArrayLike, None]
ParameterState = Mapping[Parameter, npt.NDArray[np.float64]]
RandomState = Union[int, np.random.Generator, np.random.RandomState, None]


class InvalidDistributionState(ValueError):
    """A state gives a distribution invalid parameter values."""


class Distribution(Node, ABC):
    """A probability law with pointwise log probability scores."""

    @property
    @abstractmethod
    def parameters(self) -> Mapping[str, Expr]:
        """Named parameter expressions in evaluation order."""

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        yield from self.parameters.items()

    @abstractmethod
    def log_prob(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        """Return pointwise log scores, reducing any event coordinates."""
        raise NotImplementedError


class SampleableDistribution(Distribution, ABC):
    """A probability law that can also generate random observations."""

    @abstractmethod
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.generic]:
        raise NotImplementedError


class UnivariateContinuousDistribution(Distribution, ABC):
    """A univariate continuous law with density, CDF, and quantile operations."""

    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.exp(self.log_prob(x, state=state))

    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return self.log_prob(x, state=state)

    @abstractmethod
    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError
