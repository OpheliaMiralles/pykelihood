from __future__ import annotations

import abc
import operator
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, overload

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
