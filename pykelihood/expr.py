from __future__ import annotations

import abc
import operator
from collections.abc import Iterator
from typing import Any, Union

import numpy as np
import numpy.typing as npt

PathElem = Union[str, int]
NodePath = tuple[PathElem, ...]


def ensure_node(value: Node | npt.ArrayLike) -> Node:
    if isinstance(value, Node):
        return value
    return Constant(value)


class Node(abc.ABC):
    """Base class for graph nodes."""

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        return iter(())

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

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


class Constant(Node):
    """Literal value normalized into a graph node."""

    def __init__(self, value: npt.ArrayLike):
        self.value = np.asarray(value, dtype=np.float64)

    def __call__(self) -> npt.NDArray[np.float64]:
        return self.value

    def __repr__(self) -> str:
        return f"Constant({self.value!r})"


class FunctionExpr(Node):
    """Arithmetic expression node built from other nodes."""

    def __init__(
        self,
        function,
        args: tuple[Node, ...],
        name: str,
        arg_names: tuple[PathElem, ...] | None = None,
    ) -> None:
        self.function = function
        self.args = args
        self.name = name
        self.arg_names = arg_names

    def iter_children(self) -> Iterator[tuple[PathElem, Node]]:
        for index, arg in enumerate(self.args):
            child_name = index if self.arg_names is None else self.arg_names[index]
            yield child_name, arg

    def __call__(self):
        return self.function(*(arg() for arg in self.args))

    def __repr__(self) -> str:
        return f"FunctionExpr({self.name!r}, args={self.args!r})"
