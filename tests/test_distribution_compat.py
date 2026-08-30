"""Unit tests for the distribution compatibility projection layer."""

from __future__ import annotations

import numpy as np
from scipy import stats

from pykelihood.distributions._compat import (
    CompatibilityValue,
    as_expr,
    compatibility_flattened_param_dict,
    compatibility_optimisation_param_dict,
    compatibility_param_mapping,
    distribution_leaf_nodes,
    value_projection,
)
from pykelihood.distributions.core import (
    ParameterDefault,
    ParameterInput,
    ScipyDistribution,
)
from pykelihood.expr import Constant, FunctionExpr, replace_parameters
from pykelihood.parameters import ConstantParameter, Parameter
from pykelihood.state import PositiveTransform


def _normal(
    loc: ParameterInput = None, scale: ParameterInput = None
) -> ScipyDistribution:
    """Minimal Normal over the core ScipyDistribution."""
    defaults = {
        "loc": ParameterDefault(0.0),
        "scale": ParameterDefault(1.0, PositiveTransform()),
    }
    loc_value = loc if loc is not None else Parameter(init=0.0, name="loc")
    scale_value = (
        scale
        if scale is not None
        else Parameter(init=1.0, name="scale", transform=PositiveTransform())
    )
    return ScipyDistribution(
        stats.norm, {"loc": loc_value, "scale": scale_value}, defaults=defaults
    )


def test_compatibility_value_evaluates_against_state() -> None:
    loc = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    model = _normal(loc, scale)
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}

    projection = value_projection(model.parameters["loc"], state)
    assert isinstance(projection, CompatibilityValue)
    np.testing.assert_allclose(projection.value, 2.0)

    scale_projection = value_projection(model.parameters["scale"], state)
    np.testing.assert_allclose(scale_projection.value, 3.0)


def test_compatibility_value_from_constant_returns_value() -> None:
    model = _normal(loc=2.0, scale=3.0)
    state = {}
    loc_projection = value_projection(model.parameters["loc"], state)
    np.testing.assert_allclose(loc_projection.value, 2.0)


def test_compatibility_value_fixed_parameter_becomes_constant_parameter() -> None:
    loc = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}

    fixed_projection = value_projection(loc, state, fixed=frozenset({loc}))
    assert isinstance(fixed_projection, ConstantParameter)
    np.testing.assert_allclose(fixed_projection.value, 2.0)


def test_distribution_leaf_nodes_traverses_graph() -> None:
    loc = Parameter(init=0.0, name="loc")
    scale = Parameter(init=1.0, name="scale", transform=PositiveTransform())
    model = _normal(loc, scale)

    leaves = distribution_leaf_nodes(model)
    assert leaves == {"loc": loc, "scale": scale}


def test_distribution_leaf_nodes_shares_parameter_identity() -> None:
    shared = Parameter(init=1.0)
    model = _normal(loc=shared, scale=shared)

    leaves = distribution_leaf_nodes(model)
    assert leaves == {"loc": shared, "scale": shared}


def test_replace_parameters_preserves_shared_nodes() -> None:
    alpha = Parameter(init=1.0)
    beta = Parameter(init=2.0)
    expr = FunctionExpr(
        lambda left, right: left + right, (alpha, beta), "+", ("left", "right")
    )

    replacement = Parameter(init=99.0)
    replaced = replace_parameters(expr, {alpha: replacement})
    leaves = distribution_leaf_nodes(replaced)
    assert leaves == {"left": replacement, "right": beta}


def test_compatibility_flattened_param_dict_from_state() -> None:
    loc = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    model = _normal(loc, scale)
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}

    result = compatibility_flattened_param_dict(model, state)
    assert set(result.keys()) == {"loc", "scale"}
    np.testing.assert_allclose(result["loc"].value, 2.0)
    np.testing.assert_allclose(result["scale"].value, 3.0)


def test_compatibility_optimisation_params_excludes_fixed() -> None:
    loc = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    model = _normal(loc, scale)
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}

    result = compatibility_optimisation_param_dict(model, state)
    assert set(result.keys()) == {"loc", "scale"}
    np.testing.assert_allclose(result["loc"].value, 2.0)
    np.testing.assert_allclose(result["scale"].value, 3.0)


def test_compatibility_optimisation_params_with_fixed_excludes_fixed() -> None:
    loc = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    model = _normal(loc, scale)
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}

    # When loc is fixed, it should not appear in optimisation params.
    result = compatibility_optimisation_param_dict(model, state, fixed=frozenset({loc}))
    assert "loc" not in result
    assert "scale" in result
    np.testing.assert_allclose(result["scale"].value, 3.0)


def test_as_expr_normalizes_kernel_like_to_bound_effect() -> None:
    from pykelihood.effects import BoundEffect
    from pykelihood.kernels import linear as kernel_linear

    k = kernel_linear(a=1.0, b=2.0)
    covariate = np.asarray([0.0, 1.0, 2.0])
    bound = k.with_covariate(covariate)
    expr = as_expr(bound)
    # as_expr normalizes a Kernel (which has .effect and .covariate) into
    # a BoundEffect that bridges to the expression graph.
    assert isinstance(expr, BoundEffect)


def test_as_expr_wraps_literals_as_constants() -> None:
    expr = as_expr(3.14)
    assert isinstance(expr, Constant)
    np.testing.assert_allclose(expr.value, 3.14)


def test_compatibility_param_mapping_groups_shared_parameters() -> None:
    shared = Parameter(init=1.0)
    model = _normal(loc=shared, scale=shared)
    state = {shared: np.asarray(2.0)}

    result = compatibility_param_mapping(model, state)
    assert len(result) == 1
    value, names = result[0]
    np.testing.assert_allclose(value, 2.0)
    assert set(names) == {"loc", "scale"}


def test_compatibility_param_mapping_with_only_opt() -> None:
    loc = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    model = _normal(loc, scale)
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}

    result = compatibility_param_mapping(model, state, only_opt=True)
    assert len(result) == 2
    names = {name for _, names_tuple in result for name in names_tuple}
    assert names == {"loc", "scale"}
