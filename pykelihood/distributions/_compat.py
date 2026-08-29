"""Small read-only adapters for the deprecated distribution surface."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol, Union, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from pykelihood.effects import BoundEffect, CategoricalEffect, Effect, FunctionEffect
from pykelihood.expr import Constant, Expr, FunctionExpr, Node, NodePath
from pykelihood.parameters import ConstantParameter, Parameter
from pykelihood.state import ParameterLayout


@runtime_checkable
class _KernelLike(Protocol):
    """The narrow legacy kernel shape accepted at the distribution boundary."""

    @property
    def effect(self) -> Effect: ...

    covariate: npt.ArrayLike | None


@runtime_checkable
class _ParameterOwner(Protocol):
    @property
    def parameters(self) -> Mapping[str, Node]: ...

    def _with_parameters(self, parameters: Mapping[str, Node]) -> Node: ...


def normalize_expr(value: Expr) -> Expr:
    """Convert a legacy kernel into the explicit-state expression it represents."""

    if isinstance(value, BoundEffect):
        return value
    if isinstance(value, _KernelLike):
        covariate = value.covariate
        if covariate is None:
            covariate = np.asarray(0.0, dtype=np.float64)
        return BoundEffect(value.effect, covariate)
    return value


def as_expr(value: Expr | npt.ArrayLike) -> Expr:
    if isinstance(value, Expr):
        return normalize_expr(value)
    return Constant(value)


def evaluate(
    node: Node, state: Mapping[Parameter, npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    if not isinstance(node, Expr):
        raise TypeError(f"Distribution parameter {node!r} is not evaluable.")
    return np.asarray(node.eval(state), dtype=np.float64)


def scalar_or_array(value: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return array.copy()


class CompatibilityValue:
    """Read-only value projection used by deprecated parameter accessors."""

    def __init__(
        self,
        node: Node,
        state: Mapping[Parameter, npt.NDArray[np.float64]],
        fixed: frozenset[Parameter] = frozenset(),
    ) -> None:
        self._node = node
        self._state = state
        self._fixed = fixed

    @property
    def value(self) -> float | npt.NDArray[np.float64]:
        return scalar_or_array(evaluate(self._node, self._state))

    def __call__(self) -> float | npt.NDArray[np.float64]:
        return self.value

    def __float__(self) -> float:
        return float(np.asarray(self.value))

    def __eq__(self, other: object) -> bool:
        other_value = other.value if isinstance(other, CompatibilityValue) else other
        return bool(np.all(np.asarray(self.value) == np.asarray(other_value)))

    def __repr__(self) -> str:
        return repr(self.value)

    def __getattr__(self, name: str) -> CompatibilityValue | ConstantParameter:
        for child_name, child in self._node.iter_children():
            if str(child_name) != name:
                continue
            if isinstance(child, Parameter) and child in self._fixed:
                return ConstantParameter(scalar_or_array(evaluate(child, self._state)))
            return CompatibilityValue(child, self._state, self._fixed)
        raise AttributeError(name)


CompatibilityProjection = Union[CompatibilityValue, ConstantParameter]


def value_projection(
    node: Node,
    state: Mapping[Parameter, npt.NDArray[np.float64]],
    fixed: frozenset[Parameter] = frozenset(),
) -> CompatibilityValue | ConstantParameter:
    if isinstance(node, Parameter) and node in fixed:
        return ConstantParameter(scalar_or_array(evaluate(node, state)))
    return CompatibilityValue(node, state, fixed)


def compatibility_flattened_param_dict(
    model: Node,
    state: Mapping[Parameter, npt.NDArray[np.float64]],
    fixed: frozenset[Parameter] = frozenset(),
) -> dict[str, CompatibilityProjection]:
    return {
        name: value_projection(node, state, fixed)
        for name, node in distribution_leaf_nodes(model).items()
    }


def compatibility_optimisation_params(
    model: Node,
    state: Mapping[Parameter, npt.NDArray[np.float64]],
    fixed: frozenset[Parameter] = frozenset(),
) -> tuple[CompatibilityValue, ...]:
    return tuple(
        CompatibilityValue(parameter, state, fixed)
        for parameter in ParameterLayout.from_expr(model).parameters
        if parameter not in fixed
    )


def optimisation_leaf_nodes(
    model: Node, fixed: frozenset[Parameter] = frozenset()
) -> dict[str, Parameter]:
    free_parameters = set(ParameterLayout.from_expr(model).parameters) - set(fixed)
    return {
        name: node
        for name, node in distribution_leaf_nodes(model).items()
        if isinstance(node, Parameter) and node in free_parameters
    }


def compatibility_optimisation_param_dict(
    model: Node,
    state: Mapping[Parameter, npt.NDArray[np.float64]],
    fixed: frozenset[Parameter] = frozenset(),
) -> dict[str, CompatibilityValue]:
    return {
        name: CompatibilityValue(parameter, state, fixed)
        for name, parameter in optimisation_leaf_nodes(model, fixed).items()
    }


def compatibility_param_mapping(
    model: Node,
    state: Mapping[Parameter, npt.NDArray[np.float64]],
    fixed: frozenset[Parameter] = frozenset(),
    *,
    only_opt: bool = False,
) -> list[tuple[float | npt.NDArray[np.float64], tuple[str, ...]]]:
    free_parameters = (
        set(ParameterLayout.from_expr(model).parameters) - set(fixed)
        if only_opt
        else None
    )
    mapped: list[tuple[float | npt.NDArray[np.float64], tuple[str, ...]]] = []
    positions: dict[int, int] = {}
    for name, node in distribution_leaf_nodes(model).items():
        if free_parameters is not None and (
            not isinstance(node, Parameter) or node not in free_parameters
        ):
            continue
        position = positions.get(id(node))
        if position is None:
            positions[id(node)] = len(mapped)
            mapped.append((scalar_or_array(evaluate(node, state)), (name,)))
        else:
            value, names = mapped[position]
            mapped[position] = (value, (*names, name))
    return mapped


def leaf_nodes(node: Node, path: NodePath = ()) -> Iterator[tuple[str, Node]]:
    children = tuple(node.iter_children())
    if not children:
        yield "_".join(str(part) for part in path), node
        return
    for child_name, child in children:
        yield from leaf_nodes(child, (*path, child_name))


def distribution_leaf_nodes(distribution: Node) -> dict[str, Node]:
    return dict(leaf_nodes(distribution))


def replace_nodes(
    node: Node, replacements: Mapping[int, Node], memo: dict[int, Node] | None = None
) -> Node:
    """Rebuild the supported expression graph while preserving shared nodes."""

    if id(node) in replacements:
        return replacements[id(node)]
    cache = {} if memo is None else memo
    if id(node) in cache:
        return cache[id(node)]

    if isinstance(node, BoundEffect):
        effect = replace_nodes(node.effect, replacements, cache)
        if not isinstance(effect, Effect):
            raise TypeError("A bound effect must remain an Effect.")
        rebuilt: Node = BoundEffect(effect, node.covariate)
    elif isinstance(node, FunctionExpr):
        args = tuple(replace_nodes(arg, replacements, cache) for arg in node.args)
        rebuilt = FunctionExpr(node.function, args, node.name, node.arg_names)
    elif isinstance(node, FunctionEffect):
        args = {
            name: cast(Union[Expr, Effect], replace_nodes(arg, replacements, cache))
            for name, arg in node.args.items()
        }
        rebuilt = FunctionEffect(node.function, args, node.name)
    elif isinstance(node, CategoricalEffect):
        args = {
            level: cast(Union[Expr, Effect], replace_nodes(arg, replacements, cache))
            for level, arg in node.level_args.items()
        }
        rebuilt = CategoricalEffect(node.levels, args)
    elif isinstance(node, _ParameterOwner):
        parameters = {
            name: replace_nodes(child, replacements, cache)
            for name, child in node.parameters.items()
        }
        rebuilt = node._with_parameters(parameters)
    else:
        rebuilt = node

    cache[id(node)] = rebuilt
    return rebuilt


def replace_parameters(node: Node, replacements: Mapping[Parameter, Node]) -> Node:
    return replace_nodes(
        node, {id(parameter): value for parameter, value in replacements.items()}
    )
