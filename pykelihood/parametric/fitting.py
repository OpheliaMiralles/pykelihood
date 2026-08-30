"""Core fitting machinery with deprecated compatibility projections."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize

from pykelihood.distributions._compat import (
    CompatibilityProjection,
    CompatibilityValue,
    _BoundDistribution,
    compatibility_flattened_param_dict,
    compatibility_optimisation_param_dict,
    compatibility_optimisation_params,
    compatibility_param_mapping,
    distribution_leaf_nodes,
    value_projection,
)
from pykelihood.distributions.core import Distribution, ParameterState
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import ConstantParameter, Parameter
from pykelihood.state import ParameterLayout, State


class Objective(Protocol):
    """Scalar objective minimized by :func:`fit_mle`."""

    def __call__(
        self, model: Distribution, data: npt.ArrayLike, *, state: ParameterState
    ) -> float: ...


OptimizerArgs = Mapping[str, object]
StateInput = Mapping[Parameter, npt.ArrayLike]
FixedParameters = Mapping[Parameter, npt.ArrayLike]
# Legacy scores are minimization objectives, matching ``fit`` and the old
# profiler contract. They are adapted to the core objective protocol below.
CompatibilityScore = Callable[[object, npt.ArrayLike], float]


class _ConfidenceProfiler(Protocol):
    def confidence_interval(
        self, param: str, precision: float = 1e-5
    ) -> tuple[float, float]: ...


class _ProfilerFactory(Protocol):
    def __call__(
        self,
        distribution: object,
        data: npt.ArrayLike,
        *,
        score_function: CompatibilityScore,
        single_profiling_param: str,
        inference_confidence: float,
    ) -> _ConfidenceProfiler: ...


def compatibility_objective(
    score: CompatibilityScore, fixed: Mapping[Parameter, npt.ArrayLike] | None = None
) -> Objective:
    """Adapt a legacy minimization score to the core objective protocol."""

    fixed_parameters = frozenset() if fixed is None else frozenset(fixed)

    def objective(
        model: Distribution, data: npt.ArrayLike, *, state: ParameterState
    ) -> float:
        value = score(_BoundDistribution(model, state, fixed_parameters), data)
        # Older expression-valued scale models can visit an invalid starting
        # point. The new core still rejects NaN objectives; this adapter turns
        # that legacy boundary condition into the normal optimizer penalty.
        return np.inf if np.isnan(value) else value

    return objective


def _copy_value(parameter: Parameter, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != parameter.shape:
        raise ValueError(
            f"Value for {parameter!r} has shape {array.shape}, "
            f"expected {parameter.shape}."
        )
    return array.copy()


def _copy_state(
    state: StateInput, parameters: tuple[Parameter, ...], *, name: str
) -> State:
    active = set(parameters)
    unknown = tuple(parameter for parameter in state if parameter not in active)
    if unknown:
        details = ", ".join(repr(parameter) for parameter in unknown)
        raise ValueError(
            f"{name} contains parameters not present in the model: {details}"
        )
    return {
        parameter: _copy_value(parameter, value) for parameter, value in state.items()
    }


def _scalar_objective(value: object) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise TypeError(
            "The fitting objective must return one scalar value, "
            f"got an array with shape {array.shape}."
        )
    scalar = float(array)
    if np.isnan(scalar):
        raise ValueError("The fitting objective returned NaN.")
    return scalar


def _validate_transform_domains(
    state: State, parameters: tuple[Parameter, ...]
) -> None:
    for parameter in parameters:
        if parameter.transform is None:
            continue
        with np.errstate(all="ignore"):
            transformed = parameter.transform.inverse_transform(state[parameter])
        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                f"Value for {parameter!r} is outside its transform domain."
            )


def _free_layout(model: Distribution, fixed: set[Parameter]) -> ParameterLayout:
    full_layout = ParameterLayout.from_expr(model)
    free_parameters = tuple(
        parameter for parameter in full_layout.parameters if parameter not in fixed
    )
    return ParameterLayout(
        free_parameters,
        {
            parameter: full_layout.parameter_paths[parameter]
            for parameter in free_parameters
        },
    )


@dataclass
class _FitResult:
    """Point-estimate result over a structural model and physical-value state."""

    model: Distribution
    state: State
    optimizer_layout: ParameterLayout
    optimizer_x0: npt.NDArray[np.float64]
    optimize_result: OptimizeResult
    fixed: Mapping[Parameter, npt.NDArray[np.float64]]

    def __post_init__(self) -> None:
        self.state = {
            parameter: np.asarray(value, dtype=np.float64).copy()
            for parameter, value in self.state.items()
        }
        self.optimizer_x0 = np.asarray(self.optimizer_x0, dtype=np.float64).copy()
        self.fixed = {
            parameter: np.asarray(value, dtype=np.float64).copy()
            for parameter, value in self.fixed.items()
        }

    @property
    def optimizer_x(self) -> npt.NDArray[np.float64]:
        """Return the fitted coordinates passed to SciPy's optimizer."""
        return np.asarray(self.optimize_result.x, dtype=np.float64).copy()


@dataclass
class _CompatFitResult(_FitResult):
    """Deprecated ``FitResult`` projection that carries compatibility state."""

    _compat_data: npt.NDArray[np.float64] | None = field(
        default=None, init=False, repr=False
    )
    _compat_score: CompatibilityScore | None = field(
        default=None, init=False, repr=False
    )

    def _set_compatibility(
        self, data: npt.ArrayLike, score: CompatibilityScore
    ) -> None:
        self._compat_data = np.asarray(data, dtype=np.float64).copy()
        self._compat_score = score

    @property
    def fitted(self) -> _BoundDistribution:
        """Deprecated fitted-distribution projection over ``model`` and ``state``."""
        return _BoundDistribution(self.model, self.state, self.fixed)

    @property
    def params_names(self) -> tuple[str, ...]:
        return self.fitted.params_names

    @property
    def flattened_params(self) -> tuple[CompatibilityProjection, ...]:
        return self.fitted.flattened_params

    @property
    def flattened_param_dict(self) -> dict[str, CompatibilityProjection]:
        return self.fitted.flattened_param_dict

    @property
    def optimisation_params(self) -> tuple[CompatibilityValue, ...]:
        return self.fitted.optimisation_params

    @property
    def optimisation_param_dict(self) -> dict[str, CompatibilityValue]:
        return self.fitted.optimisation_param_dict

    def param_mapping(
        self, only_opt: bool = False
    ) -> list[tuple[float | npt.NDArray[np.float64], tuple[str, ...]]]:
        return self.fitted.param_mapping(only_opt)

    def fit(
        self,
        data: npt.ArrayLike | None = None,
        x0: npt.ArrayLike | None = None,
        score: CompatibilityScore | None = None,
        scipy_args: OptimizerArgs | None = None,
        **fixed_values: object,
    ) -> _CompatFitResult:
        """Deprecated named fixed-parameter refit used by the old profiler."""
        if data is None:
            if self._compat_data is None:
                raise ValueError(
                    "A refit needs data when the result has no compatibility data."
                )
            data = self._compat_data

        values = dict(fixed_values)
        method = values.pop("method", None)
        if method is not None:
            if not isinstance(method, str):
                raise TypeError("method must be a SciPy optimizer method name.")
            options = dict(scipy_args or {})
            options.setdefault("method", method)
            scipy_args = options

        flattened = distribution_leaf_nodes(self.model)
        named_fixed: dict[Parameter, npt.NDArray[np.float64]] = {}
        for name, value in values.items():
            target = flattened.get(name)
            if target is None:
                if name in self.model.parameters:
                    raise ValueError(
                        f"Distribution parameter `{name}` is structural; "
                        "only leaf Parameter nodes can be fixed during a refit."
                    )
                raise ValueError(f"Unknown distribution parameter `{name}`.")
            if not isinstance(target, Parameter):
                raise ValueError(
                    f"Distribution parameter `{name}` is structural; "
                    "only leaf Parameter nodes can be fixed during a refit."
                )
            if isinstance(value, (CompatibilityValue, ConstantParameter)):
                value = value.value
            named_fixed[target] = np.asarray(value, dtype=np.float64).copy()

        fixed = dict(self.fixed)
        fixed.update(named_fixed)
        legacy_score = self._compat_score if score is None else score
        objective = (
            None
            if legacy_score is None
            else compatibility_objective(legacy_score, fixed)
        )
        core = fit_mle(
            self.model,
            data,
            state=self.state,
            fixed=fixed,
            x0=x0,
            objective=objective,
            scipy_args=scipy_args,
        )
        result = _CompatFitResult(
            model=core.model,
            state=core.state,
            optimizer_layout=core.optimizer_layout,
            optimizer_x0=core.optimizer_x0,
            optimize_result=core.optimize_result,
            fixed=core.fixed,
        )
        if legacy_score is not None:
            result._set_compatibility(data, legacy_score)
        return result

    def confidence_interval(
        self, param: str, alpha: float = 0.05, precision: float = 1e-5
    ) -> tuple[float, float]:
        """Deprecated profiler-backed confidence interval projection."""
        from pykelihood.profiler import Profiler

        if self._compat_data is None or self._compat_score is None:
            raise ValueError("Confidence intervals require a compatibility fit result.")
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if param not in self.flattened_param_dict:
            raise ValueError(f"Parameter {param} not found in fitted distribution.")
        profiler = cast(_ProfilerFactory, Profiler)(
            self,
            self._compat_data,
            score_function=self._compat_score,
            single_profiling_param=param,
            inference_confidence=1.0 - alpha,
        )
        return profiler.confidence_interval(param, precision=precision)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fitted, name)


FitResult = _CompatFitResult


def fit_mle(
    model: Distribution,
    data: npt.ArrayLike,
    *,
    state: StateInput | None = None,
    fixed: FixedParameters | None = None,
    x0: npt.ArrayLike | None = None,
    objective: Objective | None = None,
    scipy_args: OptimizerArgs | None = None,
) -> _FitResult:
    """Fit ``model`` by minimizing a scalar objective over its free parameters.

    ``state`` and ``fixed`` contain physical parameter values keyed by
    the actual :class:`~pykelihood.parameters.Parameter` objects in ``model``.
    ``x0`` is expressed in physical values for the free parameters. The optimizer
    receives the corresponding transformed coordinates, and the returned state is
    expressed in physical values.
    """
    data_array = np.asarray(data, dtype=np.float64).copy()
    full_layout = ParameterLayout.from_expr(model)
    initial = {
        parameter: _copy_value(parameter, parameter.init)
        for parameter in full_layout.parameters
        if parameter.init is not None
    }

    if state is not None:
        initial.update(_copy_state(state, full_layout.parameters, name="state"))

    requested_fixed = {} if fixed is None else dict(fixed)
    for parameter in requested_fixed:
        if not isinstance(parameter, Parameter):
            raise TypeError("fixed must be keyed by Parameter objects.")
    fixed = _copy_state(requested_fixed, full_layout.parameters, name="fixed")
    initial.update(fixed)
    missing = tuple(
        parameter for parameter in full_layout.parameters if parameter not in initial
    )
    if missing:
        details = ", ".join(repr(parameter) for parameter in missing)
        raise ValueError(
            f"Cannot build an initial state for uninitialized parameters: {details}"
        )
    _validate_transform_domains(initial, full_layout.parameters)

    layout = _free_layout(model, set(fixed))
    if x0 is None:
        physical_x0 = layout.flatten(initial)
    else:
        physical_x0 = np.asarray(x0, dtype=np.float64).ravel().copy()
        if physical_x0.size != layout.vector_size:
            raise ValueError(
                f"Expected {layout.vector_size} values in x0, got {physical_x0.size}."
            )
        initial.update(layout.unflatten(physical_x0))
        _validate_transform_domains(initial, full_layout.parameters)
    optimizer_x0 = layout.flatten(initial, transform=True)

    objective_fn = negative_log_likelihood if objective is None else objective

    def evaluate(values: npt.ArrayLike) -> float:
        optimizer_values = np.asarray(values, dtype=np.float64).ravel()
        current = dict(initial)
        current.update(layout.unflatten(optimizer_values, transform=True))
        return _scalar_objective(objective_fn(model, data_array, state=current))

    optimizer_options = dict(scipy_args or {})
    optimizer_options.setdefault("method", "Nelder-Mead")
    if layout.vector_size == 0:
        scipy_result = OptimizeResult(
            x=np.array([], dtype=np.float64),
            fun=evaluate(optimizer_x0),
            success=True,
            status=0,
            message="No free parameters.",
            nfev=1,
        )
        final_state = dict(initial)
    else:
        scipy_minimize = cast(Callable[..., OptimizeResult], minimize)
        scipy_result = scipy_minimize(evaluate, optimizer_x0, **optimizer_options)
        optimizer_x = np.asarray(scipy_result.x, dtype=np.float64).ravel()
        if optimizer_x.size != layout.vector_size:
            raise ValueError(
                "The optimizer returned an unexpected number of parameter values: "
                f"expected {layout.vector_size}, got {optimizer_x.size}."
            )
        final_state = dict(initial)
        final_state.update(layout.unflatten(optimizer_x, transform=True))

    return _CompatFitResult(
        model=model,
        state=final_state,
        optimizer_layout=layout,
        optimizer_x0=optimizer_x0,
        optimize_result=scipy_result,
        fixed=fixed,
    )
