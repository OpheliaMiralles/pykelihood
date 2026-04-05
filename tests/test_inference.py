from __future__ import annotations

import numpy as np
import pytest

from pykelihood import parameters
from pykelihood.state import ParameterLayout, PositiveTransform, initial_state


class BareParametrized(parameters.Parametrized):
    def __init__(self, *params: parameters.Parametrized | float, names: list[str]):
        super().__init__(*params)
        self._params_names = tuple(names)

    @property
    def params_names(self):
        return self._params_names


def test_parameter_layout_traversal_is_deterministic_and_shares_parameters() -> None:
    alpha = parameters.Parameter(2.0)
    beta = parameters.Parameter(3.0)
    expr = (alpha + beta) + alpha

    layout = ParameterLayout.from_expr(expr)

    assert layout.parameters == (alpha, beta)
    assert layout.parameter_paths == {
        alpha: (("left", "left"), ("right",)),
        beta: (("left", "right"),),
    }


def test_array_valued_parameter_flattens_and_unflattens() -> None:
    alpha = parameters.Parameter(np.array([1.0, 2.0]))
    beta = parameters.Parameter(3.0)
    expr = BareParametrized(alpha, beta, names=["alpha", "beta"])
    layout = ParameterLayout.from_expr(expr)

    state = initial_state(expr)
    np.testing.assert_allclose(layout.flatten(state), np.array([1.0, 2.0, 3.0]))

    updated = layout.unflatten(np.array([4.0, 5.0, 6.0]))

    np.testing.assert_allclose(updated[alpha], np.array([4.0, 5.0]))
    np.testing.assert_allclose(updated[beta], np.array([6.0]))


def test_two_dimensional_parameter_flattens_and_unflattens() -> None:
    alpha = parameters.Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    beta = parameters.Parameter(5.0)
    expr = BareParametrized(alpha, beta, names=["alpha", "beta"])
    layout = ParameterLayout.from_expr(expr)

    state = initial_state(expr)
    np.testing.assert_allclose(
        layout.flatten(state), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )

    updated = layout.unflatten(np.array([6.0, 7.0, 8.0, 9.0, 10.0]))

    np.testing.assert_allclose(updated[alpha], np.array([[6.0, 7.0], [8.0, 9.0]]))
    np.testing.assert_allclose(updated[beta], np.array([10.0]))


def test_parameter_layout_applies_transforms_during_flatten_and_unflatten() -> None:
    scale = parameters.Parameter(2.0, transform=PositiveTransform())
    layout = ParameterLayout.from_expr(scale)
    state = {scale: np.array(2.0)}

    np.testing.assert_allclose(layout.flatten(state), np.array([2.0]))
    np.testing.assert_allclose(
        layout.flatten(state, transform=True), np.array([np.log(2.0)])
    )

    recovered = layout.unflatten(np.array([np.log(2.0)]), transform=True)

    np.testing.assert_allclose(recovered[scale], np.array([2.0]))


def test_parameter_layout_applies_transforms_to_non_scalar_parameters() -> None:
    scale = parameters.Parameter(np.array([2.0, 3.0]), transform=PositiveTransform())
    layout = ParameterLayout.from_expr(scale)
    state = {scale: np.array([2.0, 3.0])}

    np.testing.assert_allclose(layout.flatten(state), np.array([2.0, 3.0]))
    np.testing.assert_allclose(
        layout.flatten(state, transform=True), np.log(np.array([2.0, 3.0]))
    )

    recovered = layout.unflatten(np.log(np.array([2.0, 3.0])), transform=True)

    np.testing.assert_allclose(recovered[scale], np.array([2.0, 3.0]))


def test_state_values_take_precedence_over_parameter_init() -> None:
    alpha = parameters.Parameter(1.0)
    layout = ParameterLayout.from_expr(alpha)
    state = {alpha: np.array(5.0)}

    np.testing.assert_allclose(layout.flatten(state), np.array([5.0]))


def test_parameter_layout_initial_state_requires_initial_values() -> None:
    alpha = parameters.Parameter(shape=(2,))
    with pytest.raises(ValueError, match="uninitialized"):
        initial_state(alpha)


def test_state_indexing() -> None:
    alpha = parameters.Parameter(1.0)
    state = initial_state(alpha)

    assert isinstance(state, dict)
    assert len(state) == 1
    np.testing.assert_allclose(state[alpha], np.array(1.0))
