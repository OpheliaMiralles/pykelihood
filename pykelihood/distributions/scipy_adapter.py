"""SciPy implementation of explicit-state continuous distributions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
from scipy.stats import rv_continuous

from pykelihood.distributions.core import (
    ParameterInput,
    ParameterState,
    RandomState,
    SampleableDistribution,
    UnivariateContinuousDistribution,
)
from pykelihood.expr import Constant, Expr
from pykelihood.parameters import Parameter
from pykelihood.state import Transform


@dataclass(frozen=True)
class ParameterDefault:
    """Initial value and optional transform for an omitted parameter."""

    value: npt.ArrayLike
    transform: Transform | None = None


class ScipyDistribution(UnivariateContinuousDistribution, SampleableDistribution):
    """Continuous SciPy distribution evaluated from expression parameters."""

    def __init__(
        self,
        scipy_distribution: rv_continuous,
        parameters: Mapping[str, ParameterInput],
        *,
        defaults: Mapping[str, ParameterDefault] | None = None,
    ) -> None:
        shape_names = (
            ()
            if scipy_distribution.shapes is None
            else tuple(name.strip() for name in scipy_distribution.shapes.split(","))
        )
        unknown = set(parameters) - set(shape_names) - {"loc", "scale"}
        if unknown:
            raise TypeError(
                f"Unknown distribution parameters: {', '.join(sorted(unknown))}"
            )
        for name in shape_names:
            if parameters.get(name) is None:
                raise TypeError(f"Missing required distribution parameter: {name}")

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

    def _evaluated_parameters(
        self, state: ParameterState | None
    ) -> dict[str, npt.NDArray[np.float64]]:
        parameter_state = {} if state is None else state
        return {
            name: np.asarray(parameter.eval(parameter_state), dtype=np.float64)
            for name, parameter in self.parameters.items()
        }

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.float64]:
        parameters = self._evaluated_parameters(state)
        batch_shape = np.broadcast_shapes(
            *(value.shape for value in parameters.values())
        )
        sample_shape = (
            () if size is None else (size,) if isinstance(size, int) else size
        )
        return np.asarray(
            self._scipy_distribution.rvs(
                **parameters, size=sample_shape + batch_shape, random_state=random_state
            ),
            dtype=np.float64,
        )

    def pdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.pdf(x, **self._evaluated_parameters(state)),
            dtype=np.float64,
        )

    def log_prob(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.logpdf(x, **self._evaluated_parameters(state)),
            dtype=np.float64,
        )

    def cdf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.cdf(x, **self._evaluated_parameters(state)),
            dtype=np.float64,
        )

    def ppf(
        self, q: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.asarray(
            self._scipy_distribution.ppf(q, **self._evaluated_parameters(state)),
            dtype=np.float64,
        )
