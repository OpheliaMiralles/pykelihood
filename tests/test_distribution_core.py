import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pykelihood.distributions.core import (
    ParameterDefault,
    ParameterInput,
    ScipyDistribution,
)
from pykelihood.effects import linear
from pykelihood.expr import Constant
from pykelihood.parameters import Parameter
from pykelihood.state import ParameterLayout, PositiveTransform


class Normal(ScipyDistribution):
    def __init__(
        self, loc: ParameterInput = None, scale: ParameterInput = None
    ) -> None:
        super().__init__(
            stats.norm,
            {"loc": loc, "scale": scale},
            defaults={
                "loc": ParameterDefault(0.0),
                "scale": ParameterDefault(1.0, PositiveTransform()),
            },
        )


def test_distribution_is_a_traversable_graph_node() -> None:
    loc = Parameter(init=1.0)
    distribution = Normal(loc=loc, scale=loc)

    children = tuple(distribution.iter_children())
    layout = ParameterLayout.from_expr(distribution)

    assert children == (("loc", loc), ("scale", loc))
    assert layout.parameters == (loc,)
    assert layout.parameter_paths[loc] == (("loc",), ("scale",))


def test_omitted_parameters_get_defaults_and_default_transforms() -> None:
    distribution = Normal()

    loc = distribution.parameters["loc"]
    scale = distribution.parameters["scale"]

    assert isinstance(loc, Parameter)
    assert isinstance(scale, Parameter)
    assert loc.transform is None
    assert isinstance(scale.transform, PositiveTransform)
    assert_allclose(loc.eval({}), 0.0)
    assert_allclose(scale.eval({}), 1.0)
    assert_allclose(
        ParameterLayout.from_expr(distribution).flatten({}, transform=True), [0.0, 0.0]
    )


def test_supplied_parameter_is_not_modified() -> None:
    transform = PositiveTransform()
    loc = Parameter(init=2.0, transform=transform)
    scale = Parameter(init=3.0)

    distribution = Normal(loc=loc, scale=scale)

    assert distribution.parameters["loc"] is loc
    assert loc.transform is transform
    assert distribution.parameters["scale"] is scale
    assert scale.transform is None


@pytest.mark.parametrize(
    "parameters",
    ({"a": None, "loc": None, "scale": None}, {"loc": None, "scale": None}),
)
def test_parameter_without_a_default_is_required(parameters: dict[str, None]) -> None:
    with pytest.raises(TypeError, match="Missing required distribution parameter: a"):
        ScipyDistribution(
            stats.gamma,
            parameters,
            defaults={
                "loc": ParameterDefault(0.0),
                "scale": ParameterDefault(1.0, PositiveTransform()),
            },
        )


def test_literals_become_constants() -> None:
    distribution = Normal(loc=2.0, scale=3.0)

    assert isinstance(distribution.parameters["loc"], Constant)
    assert isinstance(distribution.parameters["scale"], Constant)


def test_distribution_evaluates_against_explicit_state() -> None:
    loc = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    distribution = Normal(loc=loc, scale=scale)
    state = {loc: np.asarray(2.0), scale: np.asarray(3.0)}
    values = np.asarray([1.0, 2.0, 3.0])

    assert_allclose(
        distribution.pdf(values, state=state),
        stats.norm.pdf(values, loc=2.0, scale=3.0),
    )
    assert_allclose(
        distribution.cdf(values, state=state),
        stats.norm.cdf(values, loc=2.0, scale=3.0),
    )


def test_expression_valued_parameter_uses_the_same_state() -> None:
    offset = Parameter(init=1.0)
    distribution = Normal(loc=offset + 2.0, scale=1.0)
    state = {offset: np.asarray(3.0)}

    assert_allclose(distribution.pdf(5.0, state=state), stats.norm.pdf(5.0, loc=5.0))


def test_bound_effect_is_an_evaluable_distribution_parameter() -> None:
    slope = Parameter(init=1.0)
    loc = linear(slope=slope).with_covariate(np.asarray([0.0, 1.0, 2.0]))
    distribution = Normal(loc=loc, scale=1.0)
    state = {slope: np.asarray(2.0)}

    assert_allclose(
        distribution.pdf(np.asarray([0.0, 2.0, 4.0]), state=state), stats.norm.pdf(0.0)
    )
    assert_allclose(
        distribution.pdf(0.0, state=state),
        stats.norm.pdf(0.0, loc=np.asarray([0.0, 2.0, 4.0])),
    )
    with pytest.raises(ValueError):
        distribution.pdf(np.asarray([0.0, 2.0]), state=state)
    layout = ParameterLayout.from_expr(distribution)
    assert layout.parameters == (slope,)
    assert layout.parameter_paths[slope] == (("loc", "slope"),)


def test_sampling_accepts_a_numpy_generator() -> None:
    distribution = Normal(loc=2.0, scale=3.0)

    first = distribution.rvs(size=5, random_state=np.random.default_rng(4))
    second = stats.norm.rvs(
        loc=2.0, scale=3.0, size=5, random_state=np.random.default_rng(4)
    )

    assert_allclose(first, second)
