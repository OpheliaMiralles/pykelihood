from __future__ import annotations

import inspect
import re
from collections.abc import Collection, Hashable, Mapping, Sequence
from functools import wraps
from typing import Any, Callable, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from pykelihood.effects import (
    BoundEffect,
    CategoricalEffect,
    Effect,
    FunctionEffect,
    build_effect,
)
from pykelihood.effects import categorical as _categorical_effect
from pykelihood.effects import constant as _constant_effect
from pykelihood.effects import exp as _exp_effect
from pykelihood.effects import gaussian as _gaussian_effect
from pykelihood.effects import linear as _linear_effect
from pykelihood.expr import Constant, Expr, FunctionExpr, PathElem, require_expr
from pykelihood.parameters import (
    ConstantParameter,
    Parameter,
    Parametrized,
    ensure_parametrized,
)
from pykelihood.state import initial_state

RegressionData = Union[pd.DataFrame, npt.NDArray[np.float64]]
KernelArg = Union["Kernel", Effect, Expr, npt.ArrayLike]


class _CompatNode(Parametrized):
    """Represent an expression node through the `Parametrized` interface."""

    def __init__(
        self,
        factory: Callable[..., Effect | Expr],
        *,
        name: str,
        params: Mapping[str, Parametrized],
    ) -> None:
        self._factory = factory
        self._name = name
        self._params_names = tuple(params)
        self._params = tuple(params.values())

    @property
    def params_names(self) -> tuple[str, ...]:
        return self._params_names

    def _build_instance(self, **new_params):
        merged = dict(self.param_dict)
        merged.update(
            {name: _compat_param(value) for name, value in new_params.items()}
        )
        return type(self)(self._factory, name=self._name, params=merged)

    def to_node(self) -> Effect | Expr:
        return self._factory(*(_compat_to_node(param) for param in self._params))

    def __call__(self):
        expr = require_expr(self.to_node())
        return expr.eval(initial_state(expr))


def _compat_param(value: Any) -> Parametrized:
    """Normalize a kernel argument to a `Parametrized` child."""
    if isinstance(value, Kernel):
        return value._compat
    if isinstance(value, Parametrized):
        return value
    if isinstance(value, (Effect, Expr)):
        return _compat_from_node(value)
    return ensure_parametrized(value)


def _compat_to_node(param: Parametrized) -> Effect | Expr:
    if isinstance(param, _CompatNode):
        return param.to_node()
    return param


def _compat_from_node(node: Effect | Expr) -> Parametrized:
    """Translate an expression tree into a `Parametrized` shape."""
    if isinstance(node, Parametrized):
        return node
    if isinstance(node, Constant):
        return ConstantParameter(node.value)
    if isinstance(node, FunctionExpr):
        names = tuple(
            str(index if node.arg_names is None else node.arg_names[index])
            for index, _ in enumerate(node.args)
        )
        params = {name: _compat_from_node(arg) for name, arg in zip(names, node.args)}
        return _CompatNode(
            lambda *args: FunctionExpr(
                node.function,
                tuple(require_expr(arg) for arg in args),
                node.name,
                node.arg_names,
            ),
            name=node.name,
            params=params,
        )
    if isinstance(node, FunctionEffect):
        keys = tuple(node.args)
        params = {str(name): _compat_from_node(arg) for name, arg in node.args.items()}
        return _CompatNode(
            lambda *args: FunctionEffect(
                node.function, dict(zip(keys, args)), node.name
            ),
            name=node.name,
            params=params,
        )
    if isinstance(node, CategoricalEffect):
        used_names: dict[str, int] = {}
        names_by_level: dict[Hashable, str] = {}
        for level in node.levels:
            base_name = str(level)
            count = used_names.get(base_name, 0)
            name = base_name if count == 0 else f"{base_name}_{count}"
            used_names[base_name] = count + 1
            names_by_level[level] = name
        params = {
            names_by_level[level]: _compat_from_node(node.level_args[level])
            for level in node.levels
        }
        return _CompatNode(
            lambda *args: CategoricalEffect(
                node.levels, {level: arg for level, arg in zip(node.levels, args)}
            ),
            name="categorical",
            params=params,
        )
    raise TypeError(
        f"Unsupported node type for kernel compatibility: {type(node).__name__}"
    )


class Kernel(Parametrized):
    """
    Represents a kernel function of one covariate with parameters.
    """

    def __init__(self, effect: Effect, covariate: npt.ArrayLike | None = None) -> None:
        self.covariate = covariate
        self._compat = _compat_from_node(effect)
        self._params_names = self._compat.params_names
        self._params = self._compat.params

    @classmethod
    def from_compat(
        cls, compat: Parametrized, covariate: npt.ArrayLike | None = None
    ) -> Kernel:
        instance = cls.__new__(cls)
        instance.covariate = covariate
        instance._compat = compat
        instance._params_names = compat.params_names
        instance._params = compat.params
        return instance

    @property
    def params_names(self) -> tuple[str, ...]:
        return self._params_names

    @property
    def effect(self) -> Effect:
        return _compat_to_node(self._compat)  # pyright: ignore[reportReturnType]

    def __call__(self, x=None):
        """
        Evaluate the kernel function on the given covariate.

        Parameters
        ----------
        x : array-like, optional
            Covariate values. If not provided, uses the instance's `covariate`.

        Returns
        -------
        float
            Result of the kernel function evaluation.
        """
        covariate = self.covariate if x is None else x
        if covariate is None:
            covariate = np.array(0.0, dtype=np.float64)
        return BoundEffect(self.effect, covariate).eval({})

    def _build_instance(self, **new_params):
        compat = self._compat._build_instance(**new_params)
        return type(self).from_compat(compat, self.covariate)

    def with_covariate(self, covariate):
        """
        Create a new instance of the kernel with the given covariate.

        Parameters
        ----------
        covariate : array-like
            New covariate values.

        Returns
        -------
        Kernel
            New kernel instance with updated covariate.
        """
        return type(self).from_compat(self._compat, covariate)


def _unwrap_kernel_arg(value: KernelArg) -> Effect | Expr | npt.ArrayLike:
    if isinstance(value, Kernel):
        return value.effect
    return value


def _kernel_arguments(
    **kwargs: KernelArg | None,
) -> dict[PathElem, Effect | Expr | npt.ArrayLike]:
    return {
        name: _unwrap_kernel_arg(value)
        for name, value in kwargs.items()
        if value is not None
    }


def kernel(function: Callable[..., Any]) -> Callable[..., Kernel]:
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        raise TypeError("kernel expects a function with a covariate parameter.")
    kernel_parameters = parameters[1:]
    kernel_signature = inspect.Signature(parameters=kernel_parameters)
    parameter_names = tuple(parameter.name for parameter in kernel_parameters)

    @wraps(function)
    def build(x: npt.ArrayLike | None = None, *args: Any, **kwargs: Any) -> Kernel:
        bound = kernel_signature.bind_partial(*args, **kwargs)
        effect = build_effect(
            function,
            name=function.__name__,
            parameter_names=parameter_names,
            arguments=_kernel_arguments(**bound.arguments),
        )
        return Kernel(effect, x)

    return build


def constant(value: KernelArg = 0.0) -> Kernel:
    """
    A kernel representing a constant value.

    Parameters
    ----------
    value : float, optional
        Constant value for the kernel. Default is 0.0.
    """
    unwrapped = _unwrap_kernel_arg(value)
    if isinstance(unwrapped, (Effect, Expr)):
        effect = _constant_effect(unwrapped)
    else:
        effect = _constant_effect(Parameter(unwrapped))
    return Kernel(effect, None)


def linear(
    x: npt.ArrayLike | None = None,
    *,
    a: KernelArg | None = None,
    b: KernelArg | None = None,
) -> Kernel:
    r"""
    Linear kernel function.

    .. math::

        y = a + b \cdot x

    Parameters
    ----------
    x : array-like
        Input data.
    a : float
        Intercept of the linear function.
    b : float
        Slope of the linear function.

    Returns
    -------
    array-like
        Output of the linear kernel.
    """
    effect_kwargs = {}
    if b is not None:
        effect_kwargs["slope"] = _unwrap_kernel_arg(b)

    effect: Effect = _linear_effect(**effect_kwargs)
    if not isinstance(effect, FunctionEffect):
        raise TypeError("effects.linear is expected to return a FunctionEffect.")
    linear_effect = effect
    if a is not None:
        effect = build_effect(
            lambda x, a, b: a + b * x,
            name="linear",
            parameter_names=("a", "b"),
            arguments={"a": _unwrap_kernel_arg(a), "b": linear_effect.args["slope"]},
        )
    elif x is None:
        effect = build_effect(
            lambda x, a, b: a + b * x,
            name="linear",
            parameter_names=("a", "b"),
            arguments={
                "a": Parameter(init=0.0, name="a"),
                "b": Parameter(init=0.0, name="b"),
            },
        )
    else:
        effect = build_effect(
            lambda x, a, b: a + b * x,
            name="linear",
            parameter_names=("a", "b"),
            arguments={
                "a": Parameter(init=0.0, name="a"),
                "b": linear_effect.args["slope"],
            },
        )
    return Kernel(effect, x)


@kernel
def polynomial(
    x: npt.ArrayLike, a: Any = None, b: Any = None, c: Any = None
) -> npt.NDArray[np.float64]:
    r"""
    Polynomial kernel function.

    .. math::

        y = a + b \cdot X + c \cdot X^2

    Parameters
    ----------
    X : array-like
        Input data.
    a : float
        Constant term.
    b : float
        Coefficient for the linear term.
    c : float
        Coefficient for the quadratic term.

    Returns
    -------
    array-like
        Output of the polynomial kernel.
    """
    x_value = np.asarray(x, dtype=np.float64)
    return a + b * x_value + c * x_value**2


@kernel
def exponential(
    x: npt.ArrayLike, a: Any = None, b: Any = None
) -> npt.NDArray[np.float64]:
    r"""
    Exponential kernel function.

    .. math::

        y = \exp(a + b \cdot X)

    Parameters
    ----------
    X : array-like
        Input data.
    a : float
        Coefficient for the constant term inside the exponential.
    b : float
        Coefficient for the linear term inside the exponential.

    Returns
    -------
    array-like
        Exponential of the linear function.
    """
    return np.exp(a + b * x)


@kernel
def exponential_ratio(
    x: npt.ArrayLike, a: Any = None, b: Any = None, c: Any = None
) -> npt.NDArray[np.float64]:
    r"""
    Exponential ratio kernel function.

    .. math::

        y = c \cdot \exp\left(\frac{a \cdot X}{b}\right)

    Parameters
    ----------
    X : array-like
        Input data.
    a : float
        Numerator coefficient inside the exponential.
    b : float
        Denominator coefficient inside the exponential.
    c : float
        Scaling factor.

    Returns
    -------
    array-like
        Exponential ratio kernel output.
    """
    return c * np.exp(a * x / b)


def gaussian(
    x: npt.ArrayLike | None = None,
    *,
    mu: KernelArg | None = None,
    sigma: KernelArg | None = None,
    scaling: KernelArg | None = None,
) -> Kernel:
    r"""
    Gaussian kernel function.

    .. math::

        y = \text{scaling} \cdot \frac{1}{\sigma \sqrt{2\pi}} \cdot
        \exp\left(-\frac{(X - \mu)^2}{2 \sigma^2}\right)

    Parameters
    ----------
    X : array-like
        Input data.
    mu : float
        Mean of the Gaussian function.
    sigma : float
        Standard deviation of the Gaussian function.
    scaling : float
        Scaling factor for the output.

    Returns
    -------
    array-like
        Gaussian kernel output.
    """
    effect_kwargs = {
        name: _unwrap_kernel_arg(value)
        for name, value in {"mu": mu, "sigma": sigma, "scaling": scaling}.items()
        if value is not None
    }
    effect = _gaussian_effect(**effect_kwargs)
    return Kernel(effect, x)


@kernel
def trigonometric(
    x: npt.ArrayLike, a: Any = None, b: Any = None, c: Any = None
) -> npt.NDArray[np.float64]:
    r"""
    Trigonometric kernel function.

    .. math::

        y = a + b \cdot \cos(2\pi X) + c \cdot \sin(2\pi X)

    Parameters
    ----------
    X : array-like
        Rescaled input vector per period of interest.
    a : float
        Constant term.
    b : float
        Coefficient for the cosine term.
    c : float
        Coefficient for the sine term.

    Returns
    -------
    array-like
        Trigonometric kernel output.
    """
    x_value = np.asarray(x, dtype=np.float64)
    return a + b * np.cos(2 * np.pi * x_value) + c * np.sin(2 * np.pi * x_value)


@kernel
def hawkes(
    x: npt.NDArray, mu: Any = None, alpha: Any = None, theta: Any = None
) -> npt.NDArray[np.float64]:
    r"""
    Hawkes process with exponential kernel.

    .. math::

        \lambda(t) = \mu + \alpha \cdot \sum_{t_i < t} \exp(-\theta (t - t_i))

    Parameters
    ----------
    X : array-like
        Times of occurrence of events.
    mu : float
        Background constant intensity.
    alpha : float
        Infectivity of events.
    theta : float
        Decay term describing the decrease in intensity over time.

    Returns
    -------
    array-like
        Intensity function values at each time point.
    """
    return mu + alpha * theta * np.array(
        [np.sum(np.exp(-theta * (x[i] - x[:i]))) for i in range(len(x))]
    )


def linear_regression(
    x: RegressionData, add_intercept: bool = False, **constraints: KernelArg
) -> Kernel:
    r"""
    Linear regression of the columns in the data.

    .. math::

        y = \beta_0 + \sum_{i=1}^{n} \beta_i x_i

    Parameters
    ----------
    x : array-like or int
        The number of dimensions (int) or the data the kernel will be computed on.
        There will be one parameter for each column.
    add_intercept : bool
        If True, an intercept is added to the result.
    constraints : dict, optional
        Fixed values for the parameters of the regression. The constraints are given as
        ``beta_i=value``, where ``i`` is the index of the column starting with 1.
        If `x` is provided as a dataframe and the second column is named `cname`,
        the following constraints are equivalent: ``beta_2=2``, ``beta_cname=2``, ``cname=2``.
        The parameter ``beta_0`` constrains the value of the intercept if `add_intercept` is True.

    Returns
    -------
    float
        The linear sum computed from the input data.
    """
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    elif matrix.ndim != 2:
        raise ValueError("linear_regression expects a 1- or 2-dimensional array.")
    translated_constraints = {}
    for parameter_name, value in constraints.items():
        raw_name = parameter_name.removeprefix("beta_")
        if isinstance(x, pd.DataFrame) and raw_name in x.columns:
            index = x.columns.get_loc(raw_name)
            if not isinstance(index, int):
                raise ValueError(
                    f"Unable to resolve parameter constraint: {parameter_name}"
                )
            translated_name = f"beta_{index + 1}"
        else:
            translated_name = parameter_name
        translated_constraints[translated_name] = _unwrap_kernel_arg(value)

    if "beta_0" in translated_constraints and not add_intercept:
        raise ValueError(
            "A fixed value is given for the intercept, but `add_intercept` is not True."
        )

    parameter_names = tuple(
        f"beta_{index}"
        for index in range(0 if add_intercept else 1, matrix.shape[1] + 1)
    )

    def regression_fn(
        data: npt.ArrayLike, *resolved: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        data_array = np.asarray(data, dtype=np.float64)
        if data_array.ndim == 1:
            data_array = data_array[:, np.newaxis]
        values = [np.asarray(value, dtype=np.float64) for value in resolved]
        if add_intercept:
            intercept, coefficient_values = values[0], values[1:]
        else:
            intercept = np.array(0.0, dtype=np.float64)
            coefficient_values = values
        coefficients = np.stack(coefficient_values)
        return intercept + (coefficients * data_array).sum(axis=1)

    effect = build_effect(
        regression_fn,
        name="linear_regression",
        parameter_names=parameter_names,
        arguments=translated_constraints,
    )
    return Kernel(effect, matrix)


def exponential_linear_regression(
    x: RegressionData, add_intercept: bool = False, **constraints: KernelArg
) -> Kernel:
    r"""
    Exponential of a linear sum of the columns in the data.

    .. math::

        y = \exp\left(\beta_0 + \sum_{i=1}^{n} \beta_i x_i\right)

    Parameters
    ----------
    x : array-like or int
        The number of dimensions (int) or the data the kernel will be computed on.
        There will be one parameter for each column.
    add_intercept : bool
        If True, an intercept is added to the result.
    constraints : dict, optional
        Fixed values for the parameters of the regression. The constraints are given as
        ``beta_i=value``, where ``i`` is the index of the column starting with 1.
        If `x` is provided as a dataframe and the second column is named `cname`,
        the following constraints are equivalent: ``beta_2=2``, ``beta_cname=2``, ``cname=2``.
        The parameter ``beta_0`` constrains the value of the intercept if `add_intercept` is True.

    Returns
    -------
    float
        The linear sum computed from the input data.
    """
    regression = linear_regression(x, add_intercept=add_intercept, **constraints)
    return Kernel(_exp_effect(regression.effect), regression.covariate)


def polynomial_regression(
    x: RegressionData, degree: int | Sequence[int] = 2, **constraints: KernelArg
) -> Kernel:
    r"""
    Polynomial regression in the columns of the data.

    .. math::

        y = \sum_{i=1}^{n} \sum_{d=1}^{D_i} \beta_{i,d} x_i^d

    Parameters
    ----------
    x : array-like or int
        The number of dimensions (int) or the data the kernel will be computed on.
        There will be one parameter for each column.
    degree : int or Sequence
        The degree of the polynomial for each covariate. If an integer, the same degree is used for all.
    constraints : dict, optional
        Fixed values for the parameters of the regression. The constraints are given as
        ``beta_i_d=value``, where ``i`` is the index of the column starting with 1 and ``d`` is the degree.
        If `x` is provided as a dataframe and the second column is named `cname`,
        the following constraints are equivalent: ``beta_2_2=2``, ``beta_cname_2=2``, ``cname_2=2``.

    Returns
    -------
    float
        The polynomial regression computed from the input data.
    """
    matrix = np.asarray(x, dtype=np.float64)
    if isinstance(degree, int):
        effect_degree = int(degree)
        degrees = [effect_degree] * matrix.shape[1]
    else:
        degrees = list(degree)
        if len(degrees) != matrix.shape[1]:
            raise ValueError(
                "The number of degrees is different than the number of covariates."
            )
        effect_degree = max(degrees)

    translated_constraints = {}
    for parameter_name, value in constraints.items():
        raw_name = parameter_name.removeprefix("beta_")
        match = re.match(r"^(.+)_(\d+)$", raw_name)
        if match is None:
            raise ValueError(f"Unable to parse parameter constraint: {parameter_name}")
        column_name, power = match.groups()
        if isinstance(x, pd.DataFrame) and column_name in x.columns:
            column_index = x.columns.get_loc(column_name)
            if not isinstance(column_index, int):
                raise ValueError(
                    f"Unable to resolve parameter constraint: {parameter_name}"
                )
            translated_name = f"beta_{column_index + 1}_{power}"
        else:
            translated_name = f"beta_{int(column_name)}_{power}"
        translated_constraints[translated_name] = _unwrap_kernel_arg(value)

    for column_index, max_degree in enumerate(degrees, start=1):
        for power in range(max_degree + 1, effect_degree + 1):
            translated_constraints[f"beta_{column_index}_{power}"] = 0.0

    parameter_names = tuple(
        f"beta_{column_index}_{power}"
        for column_index, max_degree in enumerate(degrees, start=1)
        for power in range(1, max_degree + 1)
    )

    filtered_constraints = {
        name: value
        for name, value in translated_constraints.items()
        if name in parameter_names
    }

    def regression_fn(
        data: npt.ArrayLike, *resolved: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        data_array = np.asarray(data, dtype=np.float64)
        expanded_columns = [
            data_array[:, column_index] ** power
            for column_index, max_degree in enumerate(degrees)
            for power in range(1, max_degree + 1)
        ]
        expanded = np.stack(expanded_columns, axis=1)
        coefficients = np.stack(
            [np.asarray(value, dtype=np.float64) for value in resolved]
        )
        return (coefficients * expanded).sum(axis=1)

    effect = build_effect(
        regression_fn,
        name="polynomial_regression",
        parameter_names=parameter_names,
        arguments=filtered_constraints,
    )
    return Kernel(effect, x)


def categories_qualitative(
    x: Collection[Hashable], fixed_values: Mapping[Any, KernelArg] | None = None
) -> Kernel:
    """
    Kernel for qualitative (categorical) data.

    Parameters
    ----------
    x : Collection
        The qualitative data containing categorical values (e.g., strings or integers).
    fixed_values : dict, optional
        A dictionary specifying constant values for certain categories. The keys are the category names,
        and the values are the fixed parameter values.

    Returns
    -------
    Kernel
        A kernel function that assigns a parameter to each unique value in the data.

    Notes
    -----
    The kernel creates one parameter for each unique category in the data. If `fixed_values` is provided,
    the corresponding categories will use the fixed parameters instead of creating new ones.

    Examples
    --------
    >>> data = ['A', 'B', 'A', 'C']
    >>> kernel = categories_qualitative(data, fixed_values={'A': 1.0})
    """
    levels = tuple(dict.fromkeys(x))
    effect = _categorical_effect(
        levels=levels,
        fixed_values=None
        if fixed_values is None
        else {key: _unwrap_kernel_arg(value) for key, value in fixed_values.items()},
    )
    return Kernel(effect, x)  # pyright: ignore[reportArgumentType]
