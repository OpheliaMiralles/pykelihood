from __future__ import annotations

import abc
import inspect
import operator
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from functools import wraps
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from pykelihood.expr import Expr, Node, PathElem, ensure_node, require_expr
from pykelihood.parameters import Parameter

TEffect = TypeVar("TEffect", bound="Effect")


def _coerce_effect_arg(value: Any) -> Effect | Expr:
    """Normalize a helper argument to something the effect graph can store."""
    if isinstance(value, Effect):
        return value
    return require_expr(ensure_node(value))


def _evaluate_effect_arg(
    node: Effect | Expr,
    covariate: npt.ArrayLike,
    state: Mapping[Parameter, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Resolve an effect/expression child to concrete numeric values."""
    if isinstance(node, Effect):
        return node.eval(covariate, state)
    return node.eval(state)


class Effect(Node, abc.ABC):
    """Base class for covariate-dependent symbolic effects."""

    @abc.abstractmethod
    def eval(
        self,
        covariate: npt.ArrayLike,
        state: Mapping[Parameter, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def with_covariate(self, covariate: npt.ArrayLike) -> BoundEffect[Self]:
        """Bind a concrete covariate while leaving the parameter state explicit."""
        return BoundEffect(self, covariate)

    def __add__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.add(left, right),
            {"left": _coerce_effect_arg(self), "right": _coerce_effect_arg(other)},
            "+",
        )

    def __radd__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.add(left, right),
            {"left": _coerce_effect_arg(other), "right": _coerce_effect_arg(self)},
            "+",
        )

    def __sub__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.sub(left, right),
            {"left": _coerce_effect_arg(self), "right": _coerce_effect_arg(other)},
            "-",
        )

    def __rsub__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.sub(left, right),
            {"left": _coerce_effect_arg(other), "right": _coerce_effect_arg(self)},
            "-",
        )

    def __mul__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.mul(left, right),
            {"left": _coerce_effect_arg(self), "right": _coerce_effect_arg(other)},
            "*",
        )

    def __rmul__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.mul(left, right),
            {"left": _coerce_effect_arg(other), "right": _coerce_effect_arg(self)},
            "*",
        )

    def __truediv__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.truediv(left, right),
            {"left": _coerce_effect_arg(self), "right": _coerce_effect_arg(other)},
            "/",
        )

    def __rtruediv__(self, other: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.truediv(left, right),
            {"left": _coerce_effect_arg(other), "right": _coerce_effect_arg(self)},
            "/",
        )

    def __pow__(self, power: Any) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, left, right: operator.pow(left, right),
            {"left": _coerce_effect_arg(self), "right": _coerce_effect_arg(power)},
            "**",
        )

    def __neg__(self) -> FunctionEffect:
        return FunctionEffect(
            lambda _x, arg: operator.neg(arg),
            {"operand": _coerce_effect_arg(self)},
            "-",
        )


class FunctionEffect(Effect):
    """Effect backed by a Python callable and a mapping of symbolic arguments."""

    def __init__(
        self,
        function: Callable[..., Any],
        args: Mapping[PathElem, Effect | Expr],
        name: str,
    ) -> None:
        self.function = function
        self.args = dict(args)
        self.name = name

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        yield from self.args.items()

    def eval(
        self,
        covariate: npt.ArrayLike,
        state: Mapping[Parameter, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        values = [
            _evaluate_effect_arg(arg, covariate, state) for arg in self.args.values()
        ]
        return np.asarray(self.function(covariate, *values), dtype=np.float64)


def build_effect(
    function: Callable[..., Any],
    *,
    name: str,
    parameter_names: Sequence[PathElem],
    arguments: Mapping[PathElem, Effect | Expr | npt.ArrayLike] | None = None,
) -> FunctionEffect:
    """Build an effect from a callable and an explicit named argument mapping.

    ``arguments`` supplies the resolved node or literal value for each named callable
    parameter. Omitted names become free ``Parameter`` objects with ``init=0.0``.

    Helper constructors that want different defaults should create explicit
    ``Parameter(init=...)`` objects and pass them in ``arguments``.
    """
    resolved_arguments = {} if arguments is None else dict(arguments)
    final_parameters: dict[PathElem, Effect | Expr] = {}

    for parameter_name in parameter_names:
        if parameter_name in resolved_arguments:
            final_parameters[parameter_name] = _coerce_effect_arg(
                resolved_arguments[parameter_name]
            )
            continue

        if isinstance(parameter_name, str):
            final_parameters[parameter_name] = Parameter(init=0.0, name=parameter_name)
        else:
            final_parameters[parameter_name] = Parameter(init=0.0)

    return FunctionEffect(function, final_parameters, name)


class BoundEffect(Expr, Generic[TEffect]):
    """Effect with a fixed covariate and an explicit state-dependent evaluation."""

    def __init__(self, effect: TEffect, covariate: npt.ArrayLike) -> None:
        self.effect = effect
        self.covariate = covariate

    def iter_children(self):
        return self.effect.iter_children()

    def with_covariate(self, covariate: npt.ArrayLike) -> Self:
        return type(self)(self.effect, covariate)

    def eval(
        self, state: Mapping[Parameter, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        return self.effect.eval(self.covariate, state)


def define_effect(function: Callable[..., Any]) -> Callable[..., FunctionEffect]:
    """Wrap a simple effect implementation as a `FunctionEffect` builder.

    The decorated function should have signature ``(x, ...)`` where ``x`` is the
    covariate and the remaining parameters describe the effect arguments. Default
    values on those parameters are interpreted as initial values for free
    `Parameter`s created when the caller omits that argument.

    Callers of the decorated helper may pass literals, `Expr`s, or `Effect`s for
    those arguments. The decorated function itself does not receive those symbolic
    objects: at evaluation time it is called with the covariate and the resolved
    numeric array values of each argument.

    Example:
        @define_effect
        def linear(x: npt.ArrayLike, slope: npt.ArrayLike = 0.0) -> npt.NDArray[np.float64]:
            return np.asarray(slope, dtype=np.float64) * np.asarray(x, dtype=np.float64)

        effect1 = linear()  # slope is a free Parameter with init=0.0
        effect2 = linear(slope=2.0)  # slope is fixed at 2.0
        effect3 = linear(slope=Parameter(init=3.0))  # slope is a free Parameter with init=3.0
    """
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        raise TypeError("define_effect expects a function with a covariate parameter.")
    effect_parameters = parameters[1:]
    effect_signature = inspect.Signature(parameters=effect_parameters)
    parameter_names = tuple(parameter.name for parameter in effect_parameters)
    default_inits = {
        parameter.name: parameter.default
        for parameter in effect_parameters
        if parameter.default is not inspect.Parameter.empty
    }

    @wraps(function)
    def build(*args: Any, **kwargs: Any) -> FunctionEffect:
        bound = effect_signature.bind_partial(*args, **kwargs)
        arguments: dict[PathElem, Effect | Expr | npt.ArrayLike] = {
            name: value for name, value in bound.arguments.items()
        }
        for name, value in default_inits.items():
            if name not in arguments:
                arguments[name] = Parameter(init=value, name=name)
        return build_effect(
            function,
            name=function.__name__,
            parameter_names=parameter_names,
            arguments=arguments,
        )

    return build


@define_effect
def constant(x: npt.ArrayLike, value: npt.ArrayLike = 0.0) -> npt.NDArray[np.float64]:
    return np.zeros_like(x, dtype=np.float64) + np.asarray(value, dtype=np.float64)


@define_effect
def linear(x: npt.ArrayLike, slope: npt.ArrayLike = 0.0) -> npt.NDArray[np.float64]:
    x_value = np.asarray(x, dtype=np.float64)
    return np.asarray(slope, dtype=np.float64) * x_value


@define_effect
def gaussian(
    x: npt.ArrayLike,
    mu: npt.ArrayLike = 0.0,
    sigma: npt.ArrayLike = 1.0,
    scaling: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    """Return a gaussian-shaped effect over the covariate."""
    x_value = np.asarray(x, dtype=np.float64)
    mu_value = np.asarray(mu, dtype=np.float64)
    sigma_value = np.asarray(sigma, dtype=np.float64)
    scaling_value = np.asarray(scaling, dtype=np.float64)
    mult = scaling_value * 1.0 / (sigma_value * np.sqrt(2.0 * np.pi))
    expo = np.exp(-((x_value - mu_value) ** 2) / sigma_value**2)
    return mult * expo


def polynomial(
    *,
    degree: int,
    init: Mapping[int, npt.ArrayLike] | None = None,
    _coefficients: Mapping[int, Effect | Expr | npt.ArrayLike] | None = None,
) -> FunctionEffect:
    """Build a polynomial effect."""
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    parameter_names = tuple(range(degree + 1))
    named_init = {} if init is None else dict(init)
    named_coefficients = {} if _coefficients is None else dict(_coefficients)

    def polynomial_fn(
        x: npt.ArrayLike, *resolved: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        x_value = np.asarray(x, dtype=np.float64)
        broadcasted = np.broadcast_arrays(
            *[np.asarray(value, dtype=np.float64) for value in resolved]
        )
        coeffs = np.stack(broadcasted)
        return np.asarray(
            np.polynomial.polynomial.polyval(x_value, coeffs), dtype=np.float64
        )

    return build_effect(
        polynomial_fn,
        name="polynomial",
        parameter_names=parameter_names,
        arguments={
            **{name: Parameter(init=value) for name, value in named_init.items()},
            **named_coefficients,
        },
    )


class CategoricalEffect(Effect):
    """Effect that maps categorical covariate values to per-level expressions."""

    def __init__(
        self, levels: Sequence[Hashable], level_args: Mapping[Hashable, Effect | Expr]
    ) -> None:
        self.levels = tuple(levels)
        self.level_args = dict(level_args)

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        for index, level in enumerate(self.levels):
            yield index, self.level_args[level]

    def eval(
        self,
        covariate: npt.ArrayLike,
        state: Mapping[Parameter, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        values = np.asarray(covariate, dtype=object)
        mapping = {
            level: _evaluate_effect_arg(arg, covariate, state)
            for level, arg in self.level_args.items()
        }
        flattened = [mapping[value] for value in values.reshape(-1)]
        return np.asarray(flattened, dtype=np.float64).reshape(values.shape)


def categorical(
    *,
    levels: Collection[Hashable],
    fixed_values: Mapping[Any, Effect | Expr | npt.ArrayLike] | None = None,
) -> CategoricalEffect:
    """Build a categorical effect indexed by the original level values."""
    ordered_levels = tuple(dict.fromkeys(levels))
    if len(ordered_levels) != len(tuple(levels)):
        raise ValueError("categorical levels must be unique.")
    fixed_values_by_level = {} if fixed_values is None else dict(fixed_values)
    level_args: dict[Hashable, Effect | Expr] = {}
    for level in ordered_levels:
        if level in fixed_values_by_level:
            level_args[level] = _coerce_effect_arg(fixed_values_by_level[level])
        else:
            level_args[level] = Parameter(init=0.0)
    return CategoricalEffect(ordered_levels, level_args)


def exp(value: Effect | Expr | npt.ArrayLike) -> FunctionEffect:
    """Exponentiate another effect or expression."""
    return FunctionEffect(
        lambda _x, arg: np.exp(arg), {"arg": _coerce_effect_arg(value)}, "exp"
    )


def sum(
    value: Effect | Expr | npt.ArrayLike, *, axis: int | None = None
) -> FunctionEffect:
    """Reduce an effect or expression with `np.sum`."""
    return FunctionEffect(
        lambda _x, arg: np.sum(arg, axis=axis),
        {"arg": _coerce_effect_arg(value)},
        "sum",
    )
