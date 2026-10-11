"""Maximum-likelihood fitting for explicit-state distribution models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize

from pykelihood.distributions.core import Distribution, InvalidDistributionState
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.state import ParameterLayout, State

OptimizerArgs = Mapping[str, object]
StateInput = Mapping[Parameter, npt.ArrayLike]
FixedParameters = Mapping[Parameter, npt.ArrayLike]

_ModelT = TypeVar("_ModelT", bound=Distribution)


@dataclass
class FitResult(Generic[_ModelT]):
    """A structural model paired with its fitted physical-value state."""

    model: _ModelT
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
    model: _ModelT,
    data: npt.ArrayLike,
    *,
    state: StateInput | None = None,
    fixed: FixedParameters | None = None,
    scipy_args: OptimizerArgs | None = None,
) -> FitResult[_ModelT]:
    """Fit free parameters while keeping model structure unchanged.

    ``state`` and ``fixed`` values are physical values keyed by the actual
    ``Parameter`` nodes. ``state`` supplies optimizer starting values; ``fixed``
    also removes those parameters from the optimizer layout.
    """
    data_array = np.asarray(data, dtype=np.float64).copy()
    full_layout = ParameterLayout.from_node(model)
    fixed_values = {} if fixed is None else dict(fixed)
    if any(not isinstance(parameter, Parameter) for parameter in fixed_values):
        raise TypeError("fixed must be keyed by Parameter objects.")
    starting_values = {} if state is None else dict(state)
    starting_values.update(fixed_values)
    initial = full_layout.initial_state(starting_values)
    fixed_state = {parameter: initial[parameter] for parameter in fixed_values}
    layout = full_layout.without(fixed_state)
    with np.errstate(all="ignore"):
        optimizer_x0 = layout.flatten(initial, transform=True)
    for parameter in layout.parameters:
        coordinates = optimizer_x0[layout.parameter_slices[parameter]]
        if parameter.transform is not None and not np.all(np.isfinite(coordinates)):
            raise ValueError(
                f"Value for {parameter!r} is outside its transform domain."
            )

    def evaluate(values: npt.ArrayLike) -> float:
        current = dict(initial)
        current.update(layout.unflatten(values, transform=True))
        try:
            scalar = negative_log_likelihood(model, data_array, state=current)
        except InvalidDistributionState:
            return np.inf
        if np.isnan(scalar):
            return np.inf
        return scalar

    initial_objective = evaluate(optimizer_x0)
    if not np.isfinite(initial_objective):
        raise ValueError("The model has no finite likelihood at its initial state.")

    options = dict(scipy_args or {})
    options.setdefault("method", "Nelder-Mead")
    if layout.vector_size == 0:
        optimize_result = OptimizeResult(
            x=np.array([], dtype=np.float64),
            fun=initial_objective,
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
