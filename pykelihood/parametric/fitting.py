from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult

from pykelihood.likelihood import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult

from pykelihood.likelihood import log_likelihood, negative_log_likelihood
from pykelihood.state import ParameterLayout, collect_parameters, initial_state

if TYPE_CHECKING:
    from pykelihood.distributions.base import Distribution


class _ParamValueWrapper:
    """Wrapper to provide .value attribute for backward compatibility with legacy profiler."""
    def __init__(self, value):
        self.value = value


@dataclass
class FitResult:
    """Result of maximum likelihood estimation fitting."""

    model: Distribution
    data: ArrayLike
    state: dict
    optimize_result: OptimizeResult
    x0: tuple[float, ...]

    def __getattr__(self, item: str) -> Any:
        return getattr(self.model, item)

    # --- Deprecated compatibility accessors ---

    @property
    def params_names(self) -> tuple[str, ...]:
        """Deprecated: Use model.params_names instead."""
        import warnings
        warnings.warn(
            "FitResult.params_names is deprecated, use fit.model.params_names instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.model.params_names

    @property
    def flattened_params(self) -> tuple:
        """Deprecated: Use model.flattened_params instead."""
        import warnings
        warnings.warn(
            "FitResult.flattened_params is deprecated, use fit.model.flattened_params instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.model.flattened_params

    @property
    def optimisation_params(self) -> tuple["Parameter", ...]:
        """Deprecated: Use model.optimisation_params instead."""
        import warnings
        warnings.warn(
            "FitResult.optimisation_params is deprecated, use fit.model.optimisation_params instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.model.optimisation_params

    @property
    def optimisation_param_dict(self) -> dict:
        """Deprecated: Use model.optimisation_param_dict instead."""
        import warnings
        warnings.warn(
            "FitResult.optimisation_param_dict is deprecated, use fit.model.optimisation_param_dict instead",
            DeprecationWarning,
            stacklevel=2
        )
        return {name: node for name, node in self.model._get_param_nodes().items()
                if isinstance(node, Parameter)}

    def param_mapping(self) -> list[tuple[Any, tuple[str, ...]]]:
        """Deprecated: Use model.param_mapping() instead."""
        import warnings
        warnings.warn(
            "FitResult.param_mapping() is deprecated, use fit.model.param_mapping() instead",
            DeprecationWarning,
            stacklevel=2
        )
        state = self.state or {}
        nodes = self.model._get_param_nodes()
        return [
            (node.eval(state).item() if isinstance(node, Parameter) else node.eval({}), (name,))
            for name, node in nodes.items()
            if isinstance(node, Parameter)
        ]

    @property
    def flattened_param_dict(self) -> dict:
        """Deprecated: Use model.flattened_param_dict instead."""
        import warnings
        warnings.warn(
            "FitResult.flattened_param_dict is deprecated, use fit.model.flattened_param_dict instead",
            DeprecationWarning,
            stacklevel=2
        )
        state = self.state or {}
        nodes = self.model._get_param_nodes()
        result = {}
        for name, node in nodes.items():
            val = node.eval(state)
            if hasattr(val, 'item'):
                val = val.item()
            result[name] = _ParamValueWrapper(val)
        return result


def fit_mle(
    model: Distribution,
    data: ArrayLike,
    x0: ArrayLike | None = None,
    score: Callable[[Distribution, ArrayLike], float] | None = None,
    scipy_args: dict | None = None,
    **fixed_values: ArrayLike,
) -> FitResult:
    """
    Fit a distribution to data using maximum likelihood estimation.

    Parameters
    ----------
    model : Distribution
        The distribution to fit.
    data : ArrayLike
        The data to fit against.
    x0 : ArrayLike, optional
        Initial guess for the parameters.
    score : Callable, optional
        Scoring function, defaults to negative log-likelihood.
    scipy_args : dict, optional
        Additional arguments for scipy.optimize.minimize.
    fixed_values : dict
        Fixed values for specific parameters.

    Returns
    -------
    FitResult
        The result of the fit.
    """
    data = np.asarray(data, dtype=np.float64)
    score_fn = score or negative_log_likelihood

    expr = model._node
    layout = ParameterLayout.from_expr(expr)
    state = initial_state(expr)

    def to_minimize(flat_params: np.ndarray) -> float:
        new_state = layout.unflatten(flat_params)
        new_state.update({k: v for k, v in state.items() if k not in new_state})
        updated_model = model.with_state(new_state)
        return score_fn(updated_model, data)

    if x0 is None:
        x0 = layout.flatten(state)
    else:
        x0 = np.asarray(x0, dtype=np.float64)

    minimize_args = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 1500, "fatol": 1e-8},
    }
    if scipy_args:
        minimize_args.update(scipy_args)

    result = minimize(to_minimize, x0, **minimize_args)

    final_state = layout.unflatten(result.x)
    final_state.update({k: v for k, v in state.items() if k not in final_state})

    return FitResult(
        model=model.with_state(final_state),
        data=data,
        state=final_state,
        optimize_result=result,
        x0=tuple(float(v) for v in x0),
    )