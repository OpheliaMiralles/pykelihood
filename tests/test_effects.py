from __future__ import annotations

from enum import Enum, auto

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pykelihood.effects import (
    BoundEffect,
    CategoricalEffect,
    Effect,
    FunctionEffect,
    build_effect,
    categorical,
    constant,
    exp,
    gaussian,
    linear,
    polynomial,
)
from pykelihood.effects import sum as effect_sum
from pykelihood.expr import Expr
from pykelihood.parameters import Parameter
from pykelihood.state import ParameterLayout, initial_state


def test_constant_effect_evaluation() -> None:
    x = np.array([0.0, 1.0, 2.0])
    const_effect = constant(3.0)
    assert_allclose(const_effect.with_covariate(x).eval({}), np.array([3.0, 3.0, 3.0]))


def test_linear_effect_evaluation() -> None:
    x = np.array([0.0, 1.0, 2.0])
    linear_effect = linear(slope=2.0)
    assert isinstance(linear_effect, Effect)
    assert_allclose(linear_effect.with_covariate(x).eval({}), np.array([0.0, 2.0, 4.0]))


def test_effect_addition_evaluation() -> None:
    x = np.array([0.0, 1.0, 2.0])
    affine_effect = constant(1.0) + linear(slope=2.0)
    assert isinstance(affine_effect, Effect)
    assert_allclose(affine_effect.with_covariate(x).eval({}), np.array([1.0, 3.0, 5.0]))


def test_effect_helpers_create_free_parameters_when_omitted() -> None:
    linear_effect = linear()
    const_effect = constant()
    assert isinstance(linear_effect, FunctionEffect)
    assert isinstance(const_effect, FunctionEffect)

    assert isinstance(linear_effect.args["slope"], Parameter)
    assert isinstance(const_effect.args["value"], Parameter)
    assert linear_effect.args["slope"].init is not None
    assert const_effect.args["value"].init is not None
    assert_allclose(linear_effect.args["slope"].init, np.array(0.0))
    assert_allclose(const_effect.args["value"].init, np.array(0.0))


def test_build_effect_creates_free_parameters_for_omitted_arguments() -> None:
    effect = build_effect(
        lambda x, offset, slope: offset + slope * x,
        name="affine",
        parameter_names=("offset", "slope"),
        arguments={"offset": 1.0},
    )

    assert isinstance(effect.args["slope"], Parameter)
    assert effect.args["slope"].init is not None
    assert_allclose(effect.args["slope"].init, np.array(0.0))


def test_build_effect_uses_fixed_values() -> None:
    effect = build_effect(
        lambda x, a, b, c: a + b * x + c * x**2,
        name="quadratic",
        parameter_names=("a", "b", "c"),
        arguments={"a": 1.0, "b": 2.0, "c": 3.0},
    )

    assert_allclose(
        effect.with_covariate(np.array([1.0, 2.0])).eval({}), np.array([6.0, 17.0])
    )


def test_build_effect_accepts_symbolic_values() -> None:
    shift = Parameter(2.0)
    offset = shift + 1.0
    effect = build_effect(
        lambda x, offset, slope: offset + slope * x,
        name="affine",
        parameter_names=("offset", "slope"),
        arguments={"offset": offset, "slope": Parameter(init=3.0, name="slope")},
    )

    assert_allclose(
        effect.with_covariate(np.array([0.0, 1.0, 2.0])).eval({}),
        np.array([3.0, 6.0, 9.0]),
    )


def test_polynomial_effect_uses_degree_and_named_coefficients() -> None:
    x = np.array([0.0, 1.0, 2.0])
    effect = polynomial(degree=2, init={1: 2.0}, _coefficients={0: 1.0, 2: 3.0})
    assert isinstance(effect, FunctionEffect)

    assert isinstance(effect.args[1], Parameter)
    assert_allclose(effect.with_covariate(x).eval({}), 1 + 2.0 * x + 3.0 * x**2)


def test_polynomial_effect_supports_vector_coefficients() -> None:
    effect = polynomial(
        degree=2,
        _coefficients={
            0: np.array([1.0, 10.0]),
            1: np.array([2.0, 20.0]),
            2: np.array([3.0, 30.0]),
        },
    )

    assert_allclose(
        effect.with_covariate(np.array(2.0)).eval({}), np.array([17.0, 170.0])
    )


def test_effects_can_be_composed_with_math_and_reductions() -> None:
    x = np.array([0.0, 1.0, 2.0])
    base = polynomial(degree=2, _coefficients={0: 1.0, 1: 2.0, 2: 3.0})
    composed = exp(base)
    reduced = effect_sum(base)

    base_value = 1 + 2.0 * x + 3.0 * x**2
    assert_allclose(composed.with_covariate(x).eval({}), np.exp(base_value))
    assert_allclose(reduced.with_covariate(x).eval({}), np.sum(base_value))


def test_effects_can_mix_with_exprs() -> None:
    x = np.array([0.0, 1.0, 2.0])
    shift = Parameter(init=1.5)
    mixed = polynomial(degree=1, _coefficients={0: 1.0, 1: 2.0}) + shift

    assert_allclose(
        mixed.with_covariate(x).eval({shift: np.array(1.5)}), np.array([2.5, 4.5, 6.5])
    )


def test_effect_evaluation_accepts_full_explicit_state() -> None:
    x = np.array([0.0, 1.0, 2.0])
    c = Parameter()
    slope = Parameter()
    effect = constant(c) + linear(slope)
    bound = effect.with_covariate(x)
    state = {c: np.array(1.0), slope: np.array(2.0)}

    assert_allclose(bound.eval(state), np.array([1.0, 3.0, 5.0]))


def test_effect_evaluation_uses_explicitly_merged_initial_state() -> None:
    x = np.array([0.0, 1.0, 2.0])
    slope = Parameter(5.0)
    effect = constant(1.0) + linear(slope=slope)
    bound = effect.with_covariate(x)
    state = initial_state(bound) | {slope: np.array(2.0)}

    assert_allclose(bound.eval(state), np.array([1.0, 3.0, 5.0]))


def test_effect_composition_uses_nested_bound_covariate_over_outer_one() -> None:
    inner_x = np.array([0.0, 1.0, 2.0])
    outer_x = np.array([10.0, 20.0, 30.0])
    slope = Parameter()
    nested = linear(slope=slope).with_covariate(inner_x)
    effect = constant(1.0) + nested

    expected = np.array([1.0, 3.0, 5.0])
    assert_allclose(
        effect.with_covariate(outer_x).eval({slope: np.array(2.0)}), expected
    )
    assert_allclose(effect.eval(outer_x, {slope: np.array(2.0)}), expected)


def test_gaussian_effect_is_available_as_reference_builtin() -> None:
    x = np.array([0.0, 1.0])
    effect = gaussian(mu=0.0, sigma=1.0, scaling=1.0)

    assert_allclose(
        effect.with_covariate(x).eval({}),
        np.array([1.0 / np.sqrt(2.0 * np.pi), np.exp(-1.0) / np.sqrt(2.0 * np.pi)]),
    )


def test_bound_linear_effect_discovers_slope_parameter() -> None:
    x = np.array([0.0, 1.0, 2.0])
    slope = Parameter(2.0)
    effect = linear(slope=slope).with_covariate(x)
    layout = ParameterLayout.from_expr(effect)

    assert isinstance(effect, BoundEffect)
    assert layout.parameters == (slope,)
    assert layout.parameter_paths[slope] == (("slope",),)


def test_effect_layout_tracks_all_paths_for_shared_parameter() -> None:
    x = np.array([0.0, 1.0, 2.0])
    shared = Parameter(2.0)
    effect = (constant(shared) + linear(slope=shared)).with_covariate(x)
    layout = ParameterLayout.from_expr(effect)

    assert layout.parameters == (shared,)
    assert layout.parameter_paths[shared] == (("left", "value"), ("right", "slope"))
    assert_allclose(effect.eval({shared: np.array(3.0)}), np.array([3.0, 6.0, 9.0]))


def test_exponential_effect_is_expressed_with_exp_and_linear() -> None:
    x = np.array([0.0, 1.0, 2.0])
    effect = exp(constant(1.0) + linear(slope=2.0))

    assert_allclose(
        effect.with_covariate(x).eval({}), np.exp(np.array([1.0, 3.0, 5.0]))
    )


def test_linear_intercept_is_expressed_with_arithmetic() -> None:
    x = np.array([0.0, 1.0, 2.0])
    effect = constant(1.0) + linear(slope=2.0)

    assert_allclose(effect.with_covariate(x).eval({}), np.array([1.0, 3.0, 5.0]))


def test_categorical_uses_original_levels_for_matching() -> None:
    levels = [1, "1"]
    covariate = np.asarray([1, "1", 1, "1"], dtype=object)
    effect = categorical(levels=levels, fixed_values={1: 2.0, "1": 3.0})

    assert tuple(effect.level_args) == (1, "1")
    level_one = effect.level_args[1]
    level_string_one = effect.level_args["1"]
    assert isinstance(level_one, Expr)
    assert isinstance(level_string_one, Expr)
    assert_allclose(level_one.eval({}), np.array(2.0))
    assert_allclose(level_string_one.eval({}), np.array(3.0))
    assert_allclose(
        effect.with_covariate(covariate).eval({}), np.array([2.0, 3.0, 2.0, 3.0])
    )


def test_categorical_accepts_enum_levels() -> None:
    class Category(Enum):
        FIRST = auto()
        SECOND = auto()

    levels = [Category.FIRST, None, Category.SECOND]
    covariate = np.asarray(levels, dtype=object)
    effect = categorical(
        levels=levels, fixed_values={Category.FIRST: 2.0, Category.SECOND: 3.0}
    )
    assert isinstance(effect, CategoricalEffect)
    assert tuple(effect.level_args) == (Category.FIRST, None, Category.SECOND)
    none_level = effect.level_args[None]
    assert isinstance(none_level, Parameter)
    bound = effect.with_covariate(covariate)

    assert_allclose(
        bound.eval(initial_state(bound) | {none_level: np.array(4.0)}),
        np.array([2.0, 4.0, 3.0]),
    )


def test_categorical_rejects_duplicate_levels() -> None:
    with pytest.raises(ValueError, match="unique"):
        categorical(levels=[1, 2, 1])
