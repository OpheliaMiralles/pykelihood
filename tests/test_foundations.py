from typing import Union

import numpy as np
import pytest

from pykelihood import parameters
from pykelihood.expr import Constant, FunctionExpr, Node, PathElem, ensure_node


class BareParametrized(parameters.Parametrized):
    def __init__(
        self, *params: Union[parameters.Parametrized, float], names: list[str]
    ):
        super().__init__(*params)
        self._params_names = tuple(names)

    @property
    def params_names(self) -> tuple[str, ...]:
        return self._params_names


def walk_nodes(node: Node, path: tuple[PathElem, ...] = ()):
    yield path, node
    for child_name, child in node.iter_children():
        yield from walk_nodes(child, (*path, child_name))


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (lambda param: param + 3.0, 5.0),
        (lambda param: 3.0 + param, 5.0),
        (lambda param: param - 3.0, -1.0),
        (lambda param: 3.0 - param, 1.0),
        (lambda param: param * 3.0, 6.0),
        (lambda param: 3.0 * param, 6.0),
        (lambda param: param / 4.0, 0.5),
        (lambda param: 4.0 / param, 2.0),
        (lambda param: param**3.0, 8.0),
        (lambda param: -param, -2.0),
    ],
)
def test_arithmetic_operators_evaluate_as_expected(builder, expected) -> None:
    param = parameters.Parameter(2.0)

    expr = builder(param)

    assert isinstance(expr, FunctionExpr)
    np.testing.assert_allclose(expr(), expected)


def test_parameter_traversal_is_deterministic() -> None:
    alpha = parameters.Parameter(1.0)
    inner = BareParametrized(alpha, 2.0, names=["alpha", "offset"])
    outer = BareParametrized(
        inner, parameters.ConstantParameter(3.0), names=["loc", "scale"]
    )

    paths = [path for path, _ in walk_nodes(outer)]

    assert paths == [(), ("loc",), ("loc", "alpha"), ("loc", "offset"), ("scale",)]


def test_literal_values_normalize_into_constant_nodes() -> None:
    expr = parameters.Parameter(2.0) + 3.0

    children = list(expr.iter_children())

    assert isinstance(children[1][1], Constant)
    np.testing.assert_allclose(children[1][1](), 3.0)
    assert children[1][1]().dtype == np.float64


def test_ensure_node_is_idempotent_for_nodes() -> None:
    parameter = parameters.Parameter(2.0)

    assert ensure_node(parameter) is parameter


def test_ensure_node_wraps_literals_as_float64_constants() -> None:
    constant = ensure_node([1, 2, 3])

    assert isinstance(constant, Constant)
    np.testing.assert_allclose(constant(), np.array([1.0, 2.0, 3.0]))
    assert constant.value.dtype == np.float64


def test_nested_arithmetic_expressions_evaluate() -> None:
    expr = (parameters.Parameter(2.0) + 1.5) * 4.0 - 3.0

    assert isinstance(expr, FunctionExpr)
    np.testing.assert_allclose(expr(), 11.0)


def test_binary_expression_children_use_left_and_right_names() -> None:
    expr = parameters.Parameter(2.0) + parameters.Parameter(3.0)

    child_names = [name for name, _ in expr.iter_children()]

    assert child_names == ["left", "right"]


def test_unary_expression_children_use_operand_name() -> None:
    expr = -parameters.Parameter(2.0)

    child_names = [name for name, _ in expr.iter_children()]

    assert child_names == ["operand"]


def test_function_expr_without_explicit_arg_names_uses_indices() -> None:
    expr = FunctionExpr(
        lambda left, right: left + right,
        (parameters.Parameter(2.0), parameters.Parameter(3.0)),
        "custom",
    )

    child_names = [name for name, _ in expr.iter_children()]

    assert child_names == [0, 1]


def test_shared_parameters_keep_identity_in_expression_graphs() -> None:
    alpha = parameters.Parameter(2.0)
    expr = alpha + alpha

    children = list(expr.iter_children())
    traversed = list(walk_nodes(expr))

    assert children[0][1] is alpha
    assert children[1][1] is alpha
    assert traversed[1][1] is traversed[2][1]


def test_walk_nodes_traverses_expression_preorder() -> None:
    left = parameters.Parameter(2.0)
    right = parameters.Parameter(3.0)
    expr = (left + 1.0) * right

    traversal = [(path, type(node).__name__) for path, node in walk_nodes(expr)]

    assert traversal == [
        ((), "FunctionExpr"),
        (("left",), "FunctionExpr"),
        (("left", "left"), "Parameter"),
        (("left", "right"), "Constant"),
        (("right",), "Parameter"),
    ]
