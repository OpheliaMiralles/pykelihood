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
    fixed_values = {} if fixed is None else dict(fixed)
    if any(not isinstance(parameter, Parameter) for parameter in fixed_values):
        raise TypeError("fixed must be keyed by Parameter objects.")
    starting_values = {} if state is None else dict(state)
    starting_values.update(fixed_values)
    initial = full_layout.initial_state(starting_values)
    fixed_state = {parameter: initial[parameter] for parameter in fixed_values}
    _validate_transform_domains(initial, full_layout.parameters)

    layout = full_layout.without(fixed_state)
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
