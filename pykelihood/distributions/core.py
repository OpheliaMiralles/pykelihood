from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.stats import rv_continuous

from pykelihood.expr import Constant, Expr, Node, PathElem
from pykelihood.parameters import Parameter
from pykelihood.state import Transform

ParameterInput = Union[Expr, npt.ArrayLike, None]
RandomState = Union[int, np.random.Generator, np.random.RandomState, None]
ParameterState = Mapping[Parameter, npt.NDArray[np.float64]]


def _require_shape_parameters(
    scipy_distribution: rv_continuous, parameters: Mapping[str, ParameterInput]
) -> None:
    shape_names = (
        ()
        if scipy_distribution.shapes is None
        else tuple(name.strip() for name in scipy_distribution.shapes.split(","))
    )
    for name in shape_names:
        if parameters.get(name) is None:
            raise TypeError(f"Missing required distribution parameter: {name}")


@dataclass(frozen=True)
class ParameterDefault:
    """Initial value and transform for a distribution-created parameter."""

    value: npt.ArrayLike
    transform: Transform | None = None


class Distribution(Node, ABC):
    """Structural distribution node evaluated against an explicit state."""

    @property
    @abstractmethod
    def parameters(self) -> Mapping[str, Expr]:
        """Distribution parameters in their public order."""

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
    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def isf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        """Evaluate the library's density-or-mass value at ``x``."""
        raise NotImplementedError

    def sf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return 1.0 - self.cdf(x, state=state)

    def logcdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.log(self.cdf(x, state=state))

    def logsf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.log(self.sf(x, state=state))

    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.log(self.pdf(x, state=state))

    def inverse_cdf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return self.ppf(q, state=state)


class ScipyDistribution(Distribution):
    """Continuous SciPy distribution with expression-valued parameters.

    Discrete distributions such as Bernoulli adapt SciPy's ``pmf`` separately.
    """

    def __init__(
        self,
        scipy_distribution: rv_continuous,
        parameters: Mapping[str, ParameterInput],
        *,
        defaults: Mapping[str, ParameterDefault] | None = None,
    ) -> None:
        _require_shape_parameters(scipy_distribution, parameters)
        parameter_defaults = {} if defaults is None else defaults
        resolved: dict[str, Expr] = {}
        for name, value in parameters.items():
            if value is None:
                if name not in parameter_defaults:
                    raise TypeError(f"Missing required distribution parameter: {name}")
                default = parameter_defaults[name]
                resolved[name] = Parameter(
                    init=default.value, transform=default.transform, name=name
                )
            elif isinstance(value, Expr):
                resolved[name] = value
            else:
                resolved[name] = Constant(value)

        self._scipy_distribution = scipy_distribution
        self._parameters = MappingProxyType(resolved)

    @property
    def parameters(self) -> Mapping[str, Expr]:
        return self._parameters

    def _scipy_parameters(
        self, state: ParameterState | None
    ) -> dict[str, npt.NDArray[np.float64]]:
        parameter_state = {} if state is None else state
        return {
            name: parameter.eval(parameter_state)
            for name, parameter in self.parameters.items()
        }

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.rvs(
                **self._scipy_parameters(state), size=size, random_state=random_state
            ),
            dtype=np.float64,
        )

    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.cdf(x, **self._scipy_parameters(state)),
            dtype=np.float64,
        )

    def isf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.isf(q, **self._scipy_parameters(state)),
            dtype=np.float64,
        )

    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.ppf(q, **self._scipy_parameters(state)),
            dtype=np.float64,
        )

    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.pdf(x, **self._scipy_parameters(state)),
            dtype=np.float64,
        )

    def sf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.sf(x, **self._scipy_parameters(state)),
            dtype=np.float64,
        )

    def logcdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.logcdf(x, **self._scipy_parameters(state)),
            dtype=np.float64,
        )

    def logsf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.logsf(x, **self._scipy_parameters(state)),
            dtype=np.float64,
        )

    def logpdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.logpdf(x, **self._scipy_parameters(state)),
            dtype=np.float64,
        )
