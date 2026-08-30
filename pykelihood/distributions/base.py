"""Compatibility projections over the explicit-state distribution core."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, Union, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import rv_continuous

from pykelihood.distributions._compat import (
    CompatibilityValue,
    as_expr,
    compatibility_param_mapping,
    distribution_leaf_nodes,
    optimisation_leaf_nodes,
    value_projection,
)
from pykelihood.distributions.core import Distribution as CoreDistribution
from pykelihood.distributions.core import (
    ParameterDefault,
    ParameterInput,
    ParameterState,
    RandomState,
)
from pykelihood.distributions.core import ScipyDistribution as CoreScipyDistribution
from pykelihood.expr import Expr, Node, replace_parameters
from pykelihood.metrics import opposite_log_likelihood
from pykelihood.parameters import ConstantParameter, Parameter
from pykelihood.state import ParameterLayout, State

if TYPE_CHECKING:
    from pykelihood.parametric.fitting import FitResult, OptimizerArgs, StateInput


class Reparametrization(Protocol):
    """Compatibility-only conversion from public to SciPy parameter values."""

    def __call__(
        self, parameters: Mapping[str, npt.NDArray[np.float64]]
    ) -> Mapping[str, npt.ArrayLike]: ...


LegacyScore = Callable[[object, npt.ArrayLike], float]
FixedValue = Union[Expr, npt.ArrayLike]


# Core ``Distribution`` and ``ScipyDistribution`` live in ``distributions.core``.
# All deprecated compat methods live in ``_LegacyDistribution`` and
# ``_LegacyScipyDistribution`` below.  ``Distribution`` and ``ScipyDistribution``
# re-export the clean public aliases backed by the compatibility surface.
class _LegacyDistribution(CoreDistribution):
    """Explicit-state distribution with read-only legacy projections."""

    @property
    def params_names(self) -> tuple[str, ...]:
        return tuple(self.parameters)

    @property
    def flattened_param_nodes(self) -> dict[str, Node]:
        return distribution_leaf_nodes(self)

    @property
    def flattened_params(self) -> tuple[Node, ...]:
        return tuple(self.flattened_param_nodes.values())

    @property
    def flattened_param_dict(self) -> dict[str, Node]:
        return self.flattened_param_nodes

    @property
    def optimisation_params(self) -> tuple[Parameter, ...]:
        return ParameterLayout.from_expr(self).parameters

    @property
    def optimisation_param_dict(self) -> dict[str, Parameter]:
        return optimisation_leaf_nodes(self)

    def param_mapping(
        self, only_opt: bool = False
    ) -> list[tuple[float | npt.NDArray[np.float64], tuple[str, ...]]]:
        return compatibility_param_mapping(self, {}, only_opt=only_opt)

    def _with_parameters(self, parameters: Mapping[str, Node]) -> Distribution:
        """Return the same structural model with replacement top-level nodes."""

        if not hasattr(self, "_parameters"):
            raise TypeError(f"{type(self).__name__} cannot replace its parameters.")
        result = copy.copy(self)
        object.__setattr__(result, "_parameters", MappingProxyType(dict(parameters)))
        return result

    def with_params(
        self,
        params: Sequence[Expr | npt.ArrayLike] | None = None,
        **named_params: Expr | npt.ArrayLike,
    ) -> Distribution:
        """Deprecated structural replacement adapter over the expression graph."""

        if params is not None and named_params:
            raise ValueError("Please only use one way to provide values to parameters.")
        if params is not None:
            values = tuple(params)
            free_parameters = self.optimisation_params
            if len(values) > len(free_parameters):
                raise ValueError(
                    f"Expected at most {len(free_parameters)} values, got {len(values)}."
                )
            replacements = {
                parameter: as_expr(value)
                for parameter, value in zip(free_parameters, values)
            }
            return cast(Distribution, replace_parameters(self, replacements))

        return self._with_named_params(named_params)

    def _with_named_params(
        self, named_params: Mapping[str, Expr | npt.ArrayLike]
    ) -> Distribution:
        top_level: dict[str, Node] = {
            name: node for name, node in self.parameters.items()
        }
        for name in top_level:
            if name in named_params and any(
                nested.startswith(f"{name}_") for nested in named_params
            ):
                raise ValueError(
                    f"Cannot replace parameter `{name}` and one of its children together."
                )

        direct_replacements: dict[Parameter, Node] = {}
        top_replacements: dict[str, Node] = {}
        flattened = self.flattened_param_nodes
        for name, value in named_params.items():
            replacement = as_expr(value)
            if name in top_level:
                top_replacements[name] = replacement
                continue
            target = flattened.get(name)
            if target is None:
                raise ValueError(f"Unknown distribution parameter `{name}`.")
            if not isinstance(target, Parameter):
                raise ValueError(f"Distribution parameter `{name}` cannot be replaced.")  # noqa: TRY004
            direct_replacements[target] = replacement

        result: Distribution = self
        if top_replacements:
            updated: dict[str, Node] = dict(top_level)
            updated.update(top_replacements)
            result = self._with_parameters(updated)
        if direct_replacements:
            result = cast(Distribution, replace_parameters(result, direct_replacements))
        return result

    def _apply_constraints(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(data, dtype=np.float64)

    def fit(
        self,
        data: npt.ArrayLike,
        x0: npt.ArrayLike | None = None,
        score: LegacyScore | None = None,
        scipy_args: OptimizerArgs | None = None,
        *,
        state: StateInput | None = None,
        **fixed_values: FixedValue,
    ) -> FitResult:
        """Deprecated adapter from named parameters to :func:`fit_mle`."""

        return _fit_compat(
            self,
            data,
            state=state,
            x0=x0,
            score=score,
            scipy_args=scipy_args,
            fixed_values=fixed_values,
        )

    def __getattr__(self, name: str) -> Parameter | CompatibilityValue:
        try:
            parameters = object.__getattribute__(self, "_public_parameters")
        except AttributeError:
            try:
                parameters = object.__getattribute__(self, "_parameters")
            except AttributeError as error:
                raise AttributeError(name) from error
        if name in parameters:
            node = parameters[name]
            if isinstance(node, Parameter):
                return node
            return cast(CompatibilityValue, value_projection(node, {}))
        raise AttributeError(name)


class _LegacyScipyDistribution(CoreScipyDistribution, _LegacyDistribution):
    """Plain SciPy wrapper with explicit state and legacy keyword overrides."""

    def __init__(
        self,
        scipy_distribution: rv_continuous,
        parameters: Mapping[str, ParameterInput],
        *,
        defaults: Mapping[str, ParameterDefault] | None = None,
        reparametrization: Reparametrization | None = None,
    ) -> None:
        normalized = {
            name: None if value is None else as_expr(value)
            for name, value in parameters.items()
        }
        self._reparametrization = reparametrization
        super().__init__(scipy_distribution, normalized, defaults=defaults)

    def _with_parameters(
        self, parameters: Mapping[str, Node]
    ) -> _LegacyScipyDistribution:
        if not all(isinstance(parameter, Expr) for parameter in parameters.values()):
            raise TypeError("SciPy distribution parameters must be expression nodes.")
        result = copy.copy(self)
        result._parameters = MappingProxyType(
            {name: cast(Expr, parameter) for name, parameter in parameters.items()}
        )
        return result

    def _scipy_parameters(
        self, state: ParameterState | None
    ) -> dict[str, npt.NDArray[np.float64]]:
        parameters = super()._scipy_parameters(state)
        if self._reparametrization is None:
            return parameters
        return {
            name: np.asarray(value, dtype=np.float64)
            for name, value in self._reparametrization(parameters).items()
        }

    def _overridden(
        self, overrides: Mapping[str, npt.ArrayLike]
    ) -> _LegacyScipyDistribution:
        if not overrides:
            return self
        return cast(_LegacyScipyDistribution, self._with_named_params(overrides))

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.rvs(
            self._overridden(overrides), size, state=state, random_state=random_state
        )

    def cdf(
        self,
        x: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.cdf(self._overridden(overrides), x, state=state)

    def isf(
        self,
        q: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.isf(self._overridden(overrides), q, state=state)

    def ppf(
        self,
        q: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.ppf(self._overridden(overrides), q, state=state)

    def pdf(
        self,
        x: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.pdf(self._overridden(overrides), x, state=state)

    def sf(
        self,
        x: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.sf(self._overridden(overrides), x, state=state)

    def logcdf(
        self,
        x: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.logcdf(self._overridden(overrides), x, state=state)

    def logsf(
        self,
        x: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.logsf(self._overridden(overrides), x, state=state)

    def logpdf(
        self,
        x: npt.ArrayLike,
        *,
        state: ParameterState | None = None,
        **overrides: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        return CoreScipyDistribution.logpdf(self._overridden(overrides), x, state=state)


def _fit_compat(
    distribution: _LegacyDistribution,
    data: npt.ArrayLike,
    *,
    state: StateInput | None = None,
    x0: npt.ArrayLike | None = None,
    score: LegacyScore | None = None,
    scipy_args: OptimizerArgs | None = None,
    fixed_values: Mapping[str, object],
    existing_fixed: Mapping[Parameter, npt.ArrayLike] | None = None,
) -> FitResult:
    """Adapt legacy named refits to node-keyed explicit-state fitting."""

    from pykelihood.parametric.fitting import compatibility_objective, fit_mle

    values = dict(fixed_values)
    method = values.pop("method", None)
    if method is not None:
        if not isinstance(method, str):
            raise TypeError("method must be a SciPy optimizer method name.")
        options = dict(scipy_args or {})
        options.setdefault("method", method)
        scipy_args = options

    model = distribution
    top_level = dict(model.parameters)
    structural = {
        name: value
        for name, value in values.items()
        if name in top_level
        and isinstance(value, Expr)
        and not isinstance(value, ConstantParameter)
    }
    if structural:
        model = model._with_named_params(structural)
        for name in structural:
            values.pop(name)

    named_fixed: State = {}
    for name, value in values.items():
        flattened = model.flattened_param_nodes
        target = flattened.get(name)
        if isinstance(target, Parameter):
            named_fixed[target] = np.asarray(
                value.value if isinstance(value, ConstantParameter) else value,
                dtype=np.float64,
            )
            continue
        if name in model.parameters or target is not None:
            model = model._with_named_params({name: cast(FixedValue, value)})
            continue
        raise ValueError(f"Unknown distribution parameter `{name}`.")

    layout = ParameterLayout.from_expr(model)
    active = set(layout.parameters)
    projected_state = (
        None
        if state is None
        else {
            parameter: value
            for parameter, value in state.items()
            if parameter in active
        }
    )
    fixed = {
        parameter: np.asarray(value, dtype=np.float64)
        for parameter, value in (existing_fixed or {}).items()
        if parameter in active
    }
    fixed.update(named_fixed)
    constrained_data = model._apply_constraints(data)
    legacy_score = opposite_log_likelihood if score is None else score
    result = fit_mle(
        model,
        constrained_data,
        state=projected_state,
        fixed=fixed,
        x0=x0,
        objective=compatibility_objective(legacy_score),
        scipy_args=scipy_args,
    )
    result._set_compatibility(constrained_data, legacy_score)
    return result


__all__ = ["Distribution", "ScipyDistribution"]


class Distribution(_LegacyDistribution):
    """Public distribution alias preserving the original class name."""


class ScipyDistribution(_LegacyScipyDistribution):
    """Public SciPy distribution alias preserving the original class name."""
