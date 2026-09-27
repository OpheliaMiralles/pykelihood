"""Discrete distributions for explicit-state models."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
from scipy.special import xlog1py, xlogy

from pykelihood.distributions.core import (
    Distribution,
    InvalidDistributionState,
    ParameterState,
    RandomState,
)
from pykelihood.expr import Constant, Expr


class Bernoulli(Distribution):
    """Bernoulli law with probability parameter ``p`` and integer samples."""

    def __init__(self, p: Expr | npt.ArrayLike) -> None:
        self._parameters = MappingProxyType(
            {"p": p if isinstance(p, Expr) else Constant(p)}
        )

    @property
    def p(self) -> Expr:
        return self._parameters["p"]

    @property
    def parameters(self) -> Mapping[str, Expr]:
        return self._parameters

    def _probability(self, state: ParameterState | None) -> npt.NDArray[np.float64]:
        probability = np.asarray(
            self.p.eval({} if state is None else state), dtype=np.float64
        )
        if np.any(~np.isfinite(probability)) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise InvalidDistributionState("Bernoulli probability must be in [0, 1].")
        return probability

    def log_prob(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        values = np.asarray(x)
        probability = self._probability(state)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = xlogy(values, probability) + xlog1py(1 - values, -probability)
        return np.asarray(
            np.where((values == 0) | (values == 1), scores, -np.inf), dtype=np.float64
        )

    def pmf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.exp(self.log_prob(x, state=state))

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.int64]:
        probability = self._probability(state)
        sample_shape = (
            () if size is None else (size,) if isinstance(size, int) else size
        )
        rng = (
            random_state
            if isinstance(random_state, (np.random.Generator, np.random.RandomState))
            else np.random.default_rng(random_state)
        )
        return np.asarray(
            rng.binomial(1, probability, size=sample_shape + probability.shape)
        )
