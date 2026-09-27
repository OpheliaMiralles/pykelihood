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


class Distribution(Node, ABC):
    """A probability law whose parameter expressions form a graph node."""

    @property
    @abstractmethod
    def parameters(self) -> Mapping[str, Expr]:
        """Named parameter expressions in evaluation order."""

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        yield from self.parameters.items()

    @abstractmethod
    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

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
