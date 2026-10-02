"""Opt-in extreme-value distributions with statistical shape conventions."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.stats import rv_continuous

from pykelihood.distributions.core import (
    Distribution,
    ParameterInput,
    ParameterState,
    RandomState,
)
from pykelihood.distributions.scipy_adapter import ScipyDistribution
from pykelihood.expr import Constant, Expr
from pykelihood.parameters import Parameter
from pykelihood.state import PositiveTransform, Transform


def _resolve_parameter(
    value: ParameterInput,
    *,
    name: str,
    default: float,
    transform: Transform | None = None,
) -> Expr:
    if value is None:
        return Parameter(init=default, transform=transform, name=name)
    if isinstance(value, Expr):
        return value
    return Constant(value)


class _ShapeConvenience(Distribution):
    """Map one public statistical shape convention to SciPy's ``c``."""

    _scipy_distribution: rv_continuous
    _scipy_shape_sign: float

    def __init__(
        self,
        loc: ParameterInput = None,
        scale: ParameterInput = None,
        shape: ParameterInput = None,
    ) -> None:
        self._parameters = MappingProxyType(
            {
                "loc": _resolve_parameter(loc, name="loc", default=0.0),
                "scale": _resolve_parameter(
                    scale, name="scale", default=1.0, transform=PositiveTransform()
                ),
                "shape": _resolve_parameter(shape, name="shape", default=0.0),
            }
        )
        scipy_shape = self._scipy_shape_sign * self._parameters["shape"]
        self._scipy_model = ScipyDistribution(
            self._scipy_distribution,
            {
                "c": scipy_shape,
                "loc": self._parameters["loc"],
                "scale": self._parameters["scale"],
            },
        )

    @property
    def parameters(self) -> Mapping[str, Expr]:
        """Public statistical parameters, including the unmodified shape node."""
        return self._parameters

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.float64]:
        return self._scipy_model.rvs(size=size, state=state, random_state=random_state)

    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return self._scipy_model.pdf(x, state=state)

    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return self._scipy_model.logpdf(x, state=state)

    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return self._scipy_model.cdf(x, state=state)

    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return self._scipy_model.ppf(q, state=state)


class GEV(_ShapeConvenience):
    """Generalized extreme value law with ``shape = -scipy.stats.genextreme.c``."""

    _scipy_distribution = stats.genextreme
    _scipy_shape_sign = -1.0


class GPD(_ShapeConvenience):
    """Generalized Pareto law with ``shape = scipy.stats.genpareto.c``."""

    _scipy_distribution = stats.genpareto
    _scipy_shape_sign = 1.0


__all__ = ["GEV", "GPD"]
