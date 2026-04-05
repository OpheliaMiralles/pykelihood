from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from pykelihood.expr import Node, NodePath

if TYPE_CHECKING:
    from pykelihood.parameters import Parameter

State = dict["Parameter", npt.NDArray[np.float64]]


class Transform(Protocol):
    def transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    def inverse_transform(
        self, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


class IdentityTransform:
    def transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x

    def inverse_transform(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return y


class PositiveTransform:
    def transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.exp(x)

    def inverse_transform(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.log(y)


class ProbabilityTransform:
    def transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-x))

    def inverse_transform(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.log(y / (1.0 - y))


def collect_parameter_paths(expr: Node) -> dict[Parameter, tuple[NodePath, ...]]:
    from pykelihood.parameters import ConstantParameter, Parameter

    parameter_paths: defaultdict[Parameter, list[NodePath]] = defaultdict(list)

    def visit(node: Node, path: NodePath) -> None:
        if isinstance(node, ConstantParameter):
            return
        if isinstance(node, Parameter):
            parameter_paths[node].append(path)
            return
        for step, child in node.iter_children():
            visit(child, path + (step,))

    visit(expr, ())
    return {parameter: tuple(paths) for parameter, paths in parameter_paths.items()}


def collect_parameters(expr: Node) -> tuple[Parameter, ...]:
    parameter_paths = collect_parameter_paths(expr)
    return tuple(parameter_paths)


class ParameterLayout:
    def __init__(
        self,
        parameters: tuple[Parameter, ...],
        parameter_paths: Mapping[Parameter, tuple[NodePath, ...]],
    ) -> None:
        self.parameters = parameters
        self.parameter_paths = dict(parameter_paths)
        self.parameter_slices: dict[Parameter, slice] = {}
        offset = 0
        for parameter in self.parameters:
            self.parameter_slices[parameter] = slice(offset, offset + parameter.size)
            offset += parameter.size
        self.vector_size = offset

    @classmethod
    def from_expr(cls, expr: Node) -> ParameterLayout:
        parameter_paths = collect_parameter_paths(expr)
        return cls(tuple(parameter_paths), parameter_paths)

    def flatten(
        self,
        state: Mapping[Parameter, npt.NDArray[np.float64]],
        *,
        transform: bool = False,
    ) -> npt.NDArray[np.float64]:
        chunks: list[npt.NDArray[np.float64]] = []
        for parameter in self.parameters:
            if parameter in state:
                value = state[parameter]
            elif parameter.init is not None:
                value = parameter.init
            else:
                raise ValueError(
                    f"Missing value for uninitialized parameter {parameter!r}"
                )
            if transform and parameter.transform is not None:
                value = parameter.transform.inverse_transform(value)
            chunks.append(value.ravel())
        if not chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks)

    def unflatten(self, values: npt.ArrayLike, *, transform: bool = False) -> State:
        vector = np.asarray(values, dtype=np.float64).ravel()
        if vector.size != self.vector_size:
            raise ValueError(f"Expected {self.vector_size} values, got {vector.size}")

        restored: dict[Parameter, npt.NDArray[np.float64]] = {}
        offset = 0
        for parameter in self.parameters:
            size = parameter.size
            chunk = vector[offset : offset + size].reshape(parameter.shape)
            if transform and parameter.transform is not None:
                chunk = parameter.transform.transform(chunk)
            restored[parameter] = chunk
            offset += size
        return restored


def initial_state(expr: Node) -> State:
    values: dict[Parameter, npt.NDArray[np.float64]] = {}
    missing: list[Parameter] = []
    for parameter in collect_parameters(expr):
        if parameter.init is None:
            missing.append(parameter)
        else:
            values[parameter] = parameter.init
    if missing:
        details = ", ".join(repr(parameter) for parameter in missing)
        raise ValueError(
            f"Cannot build an initial state for uninitialized parameters: {details}"
        )
    return values
