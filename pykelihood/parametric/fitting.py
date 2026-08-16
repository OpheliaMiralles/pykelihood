from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Protocol, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize

from pykelihood.distributions.core import Distribution, ParameterState
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.state import ParameterLayout, State


class Objective(Protocol):
    """Scalar objective minimized by :func:`fit_mle`."""

    def __call__(
        self, model: Distribution, data: npt.ArrayLike, *, state: ParameterState
    ) -> float: ...


OptimizerArgs = Mapping[str, object]
StateInput = Mapping[Parameter, npt.ArrayLike]
FixedParameters = Mapping[Parameter, npt.ArrayLike]


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
class FitResult:
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


def fit_mle(
    model: Distribution,
    data: npt.ArrayLike,
    *,
    state: StateInput | None = None,
    fixed: FixedParameters | None = None,
    x0: npt.ArrayLike | None = None,
    objective: Objective | None = None,
    scipy_args: OptimizerArgs | None = None,
) -> FitResult:
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

    return FitResult(
        model=model,
        state=final_state,
        optimizer_layout=layout,
        optimizer_x0=optimizer_x0,
        optimize_result=scipy_result,
        fixed=fixed,
    )
