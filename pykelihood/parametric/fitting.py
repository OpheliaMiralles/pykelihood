"""Maximum-likelihood fitting for explicit-state distribution models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize

from pykelihood.distributions.core import Distribution
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.state import ParameterLayout, State

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


def _free_layout(
    full_layout: ParameterLayout, fixed: set[Parameter]
) -> ParameterLayout:
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
    """A structural model paired with its fitted physical-value state."""

    model: Distribution
    state: State
    fixed: Mapping[Parameter, npt.NDArray[np.float64]]
    optimize_result: OptimizeResult

    def __post_init__(self) -> None:
        self.state = {
            parameter: np.asarray(value, dtype=np.float64).copy()
            for parameter, value in self.state.items()
        }
        self.fixed = {
            parameter: np.asarray(value, dtype=np.float64).copy()
            for parameter, value in self.fixed.items()
        }


def fit_mle(
    model: Distribution,
    data: npt.ArrayLike,
    *,
    state: StateInput | None = None,
    fixed: FixedParameters | None = None,
    scipy_args: OptimizerArgs | None = None,
) -> FitResult:
    """Fit free parameters while keeping model structure unchanged.

    ``state`` and ``fixed`` values are physical values keyed by the actual
    ``Parameter`` nodes. ``state`` supplies optimizer starting values; ``fixed``
    also removes those parameters from the optimizer layout.
    """
    data_array = np.asarray(data, dtype=np.float64).copy()
    full_layout = ParameterLayout.from_expr(model)
    initial: State = {
        parameter: _copy_value(parameter, parameter.init)
        for parameter in full_layout.parameters
        if parameter.init is not None
    }
    if state is not None:
        initial.update(_copy_state(state, full_layout.parameters, name="state"))

    requested_fixed = {} if fixed is None else dict(fixed)
    if any(not isinstance(parameter, Parameter) for parameter in requested_fixed):
        raise TypeError("fixed must be keyed by Parameter objects.")
    fixed_state = _copy_state(requested_fixed, full_layout.parameters, name="fixed")
    initial.update(fixed_state)

    missing = tuple(
        parameter for parameter in full_layout.parameters if parameter not in initial
    )
    if missing:
        details = ", ".join(repr(parameter) for parameter in missing)
        raise ValueError(
            f"Cannot build an initial state for uninitialized parameters: {details}"
        )
    _validate_transform_domains(initial, full_layout.parameters)

    layout = _free_layout(full_layout, set(fixed_state))
    optimizer_x0 = layout.flatten(initial, transform=True)

    def evaluate(values: npt.ArrayLike) -> float:
        current = dict(initial)
        current.update(layout.unflatten(values, transform=True))
        scalar = negative_log_likelihood(model, data_array, state=current)
        if np.isnan(scalar):
            raise ValueError("The fitting objective returned NaN.")
        return scalar

    options = dict(scipy_args or {})
    options.setdefault("method", "Nelder-Mead")
    if layout.vector_size == 0:
        optimize_result = OptimizeResult(
            x=np.array([], dtype=np.float64),
            fun=evaluate(optimizer_x0),
            success=True,
            status=0,
            message="No free parameters.",
            nfev=1,
        )
    else:
        scipy_minimize = cast(Callable[..., OptimizeResult], minimize)
        optimize_result = scipy_minimize(evaluate, optimizer_x0, **options)

    optimizer_x = np.asarray(optimize_result.x, dtype=np.float64).ravel()
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
        fixed=fixed_state,
        optimize_result=optimize_result,
    )
