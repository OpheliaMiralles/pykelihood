from __future__ import annotations

import abc
import operator
from collections.abc import Iterator, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pykelihood.parameters import Parameter

PathElem = Union[str, int]
NodePath = tuple[PathElem, ...]
TNode = TypeVar("TNode", bound="Node")


@overload
def ensure_node(value: TNode) -> TNode: ...
@overload
def ensure_node(value: npt.ArrayLike) -> Constant: ...
def ensure_node(value: Node | npt.ArrayLike) -> Node:
    if isinstance(value, Node):
        return value
    return Constant(value)


def require_expr(node: Node) -> Expr:
    if not isinstance(node, Expr):
        raise TypeError("Expected an Expr.")
    return node


class Node:
    """Base class for graph nodes."""

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        return iter(())

    def __add__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.add, (self, ensure_node(other)), "+", ("left", "right")
        )

    def __radd__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.add, (ensure_node(other), self), "+", ("left", "right")
        )

    def __sub__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.sub, (self, ensure_node(other)), "-", ("left", "right")
        )

    def __rsub__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.sub, (ensure_node(other), self), "-", ("left", "right")
        )

    def __mul__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.mul, (self, ensure_node(other)), "*", ("left", "right")
        )

    def __rmul__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.mul, (ensure_node(other), self), "*", ("left", "right")
        )

    def __truediv__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.truediv, (self, ensure_node(other)), "/", ("left", "right")
        )

    def __rtruediv__(self, other: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.truediv, (ensure_node(other), self), "/", ("left", "right")
        )

    def __pow__(self, power: Any) -> FunctionExpr:
        return FunctionExpr(
            operator.pow, (self, ensure_node(power)), "**", ("left", "right")
        )

    def __neg__(self) -> FunctionExpr:
        return FunctionExpr(operator.neg, (self,), "-", ("operand",))


class Expr(Node, abc.ABC):
    """Base class for deterministic evaluable nodes."""

    @abc.abstractmethod
    def eval(
        self, state: Mapping[Parameter, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError


class Constant(Expr):
    """Literal value normalized into a graph node."""

    def __init__(self, value: npt.ArrayLike):
        self.value = np.asarray(value, dtype=np.float64)

    def eval(
        self, state: Mapping[Parameter, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        return self.value

    def __repr__(self) -> str:
        return f"Constant({self.value!r})"


class FunctionExpr(Expr):
    """Arithmetic expression node built from other nodes."""

    def __init__(
        self,
        function,
        args: tuple[Node, ...],
        name: str,
        arg_names: tuple[PathElem, ...] | None = None,
    ) -> None:
        self.function = function
        self.args = tuple(require_expr(arg) for arg in args)
        self.name = name
        self.arg_names = arg_names

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        for index, arg in enumerate(self.args):
            child_name = index if self.arg_names is None else self.arg_names[index]
            yield child_name, arg

    def eval(
        self, state: Mapping[Parameter, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        return self.function(*(arg.eval(state) for arg in self.args))

    def __repr__(self) -> str:
        return f"FunctionExpr({self.name!r}, args={self.args!r})"


@runtime_checkable
class _ParameterOwner(Protocol):
    @property
    def parameters(self) -> Mapping[str, Node]: ...

    def _with_parameters(self, parameters: Mapping[str, Node]) -> Node: ...


def replace_nodes(
    node: Node, replacements: Mapping[int, Node], memo: dict[int, Node] | None = None
) -> Node:
    """Rebuild the supported expression graph while preserving shared nodes."""

    if id(node) in replacements:
        return replacements[id(node)]
    cache = {} if memo is None else memo
    if id(node) in cache:
        return cache[id(node)]

    from pykelihood.effects import (
        BoundEffect,
        CategoricalEffect,
        Effect,
        FunctionEffect,
    )

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
            name: replace_nodes(arg, replacements, cache)
            for name, arg in node.args.items()
        }
        rebuilt = FunctionEffect(node.function, args, node.name)
    elif isinstance(node, CategoricalEffect):
        args = {
            level: replace_nodes(arg, replacements, cache)
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
