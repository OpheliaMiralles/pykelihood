import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pykelihood.distributions.core import Normal, ParameterDefault, ScipyDistribution
from pykelihood.effects import linear
from pykelihood.expr import Constant
from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.parametric import FitResult, fit_mle
from pykelihood.state import ParameterLayout, PositiveTransform


def test_distribution_children_expose_shared_parameters_to_traversal() -> None:
    location = Parameter(init=1.0)
    distribution = Normal(loc=location, scale=location)

    layout = ParameterLayout.from_expr(distribution)

    assert tuple(distribution.iter_children()) == (
        ("loc", location),
        ("scale", location),
    )
    assert layout.parameters == (location,)
    assert layout.parameter_paths[location] == (("loc",), ("scale",))


def test_omitted_optional_parameters_have_free_defaults_and_transforms() -> None:
    distribution = Normal()
    location = distribution.parameters["loc"]
    scale = distribution.parameters["scale"]

    assert isinstance(location, Parameter)
    assert location.init == 0.0
    assert location.transform is None
    assert isinstance(scale, Parameter)
    assert scale.init == 1.0
    assert isinstance(scale.transform, PositiveTransform)
    assert_allclose(distribution.pdf(0.0), stats.norm.pdf(0.0))


def test_supplied_parameter_keeps_identity_and_transform() -> None:
    transform = PositiveTransform()
    location = Parameter(init=2.0, transform=transform)
    scale = Parameter(init=3.0)

    distribution = Normal(loc=location, scale=scale)

    assert distribution.parameters["loc"] is location
    assert location.transform is transform
    assert distribution.parameters["scale"] is scale
    assert scale.transform is None


def test_literal_values_become_structural_constants() -> None:
    distribution = Normal(loc=2.0, scale=3.0)

    assert isinstance(distribution.parameters["loc"], Constant)
    assert isinstance(distribution.parameters["scale"], Constant)
    assert ParameterLayout.from_expr(distribution).parameters == ()

    location = Parameter(init=0.0)
    mixed_distribution = Normal(loc=location, scale=3.0)
    state = {location: np.asarray(2.0)}
    assert_allclose(
        mixed_distribution.pdf(2.0, state=state),
        stats.norm.pdf(2.0, loc=2.0, scale=3.0),
    )


def test_constructor_values_are_independent_of_input_arrays() -> None:
    literal = np.array(2.0)
    initial = np.array(3.0)
    distribution = Normal(loc=literal, scale=Parameter(init=initial))
    literal[...] = 8.0
    initial[...] = 9.0

    assert_allclose(distribution.pdf(2.0), stats.norm.pdf(2.0, loc=2.0, scale=3.0))


def test_shape_parameters_remain_required() -> None:
    with pytest.raises(TypeError, match="Missing required distribution parameter: a"):
        ScipyDistribution(
            stats.gamma,
            {"a": None, "loc": None, "scale": None},
            defaults={
                "loc": ParameterDefault(0.0),
                "scale": ParameterDefault(1.0, PositiveTransform()),
            },
        )


def test_unknown_scipy_parameter_is_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="Unknown distribution parameters: typo"):
        ScipyDistribution(stats.norm, {"typo": 1.0})


def test_evaluation_uses_explicit_state_and_supports_broadcasting() -> None:
    location = Parameter(init=0.0)
    scale = Parameter(init=1.0, transform=PositiveTransform())
    distribution = Normal(loc=location, scale=scale)
    values = np.array([[0.0], [1.0]])
    state = {location: np.asarray([0.0, 2.0]), scale: np.asarray(2.0)}

    actual = distribution.pdf(values, state=state)
    expected = stats.norm.pdf(values, loc=np.array([0.0, 2.0]), scale=2.0)

    assert actual.shape == (2, 2)
    assert_allclose(actual, expected)
    assert_allclose(distribution.logpdf(values, state=state), np.log(expected))
    assert_allclose(
        distribution.cdf(values, state=state),
        stats.norm.cdf(values, loc=[0.0, 2.0], scale=2.0),
    )


def test_expression_and_bound_effect_parameters_share_explicit_state() -> None:
    offset = Parameter(init=1.0)
    expression_distribution = Normal(loc=offset + 2.0, scale=1.0)
    state = {offset: np.asarray(3.0)}
    assert_allclose(
        expression_distribution.pdf(5.0, state=state), stats.norm.pdf(5.0, loc=5.0)
    )

    slope = Parameter(init=1.0)
    bound_location = linear(slope=slope).with_covariate(np.array([0.0, 1.0, 2.0]))
    effect_distribution = Normal(loc=bound_location, scale=1.0)
    effect_state = {slope: np.asarray(2.0)}
    assert_allclose(
        effect_distribution.pdf(np.array([0.0, 2.0, 4.0]), state=effect_state),
        stats.norm.pdf(0.0),
    )
    assert_allclose(
        effect_distribution.pdf(0.0, state=effect_state),
        stats.norm.pdf(0.0, loc=np.array([0.0, 2.0, 4.0])),
    )
    with pytest.raises(ValueError):
        effect_distribution.pdf(np.array([0.0, 2.0]), state=effect_state)
    assert ParameterLayout.from_expr(effect_distribution).parameters == (slope,)


def test_sampling_accepts_seeded_generator() -> None:
    distribution = Normal(loc=2.0, scale=3.0)

    actual = distribution.rvs(size=5, random_state=np.random.default_rng(4))
    expected = stats.norm.rvs(
        loc=2.0, scale=3.0, size=5, random_state=np.random.default_rng(4)
    )

    assert_allclose(actual, expected)


def test_fit_mle_uses_fixed_identity_and_returns_physical_state_without_mutation() -> (
    None
):
    model = Normal()
    location = model.parameters["loc"]
    scale = model.parameters["scale"]
    assert isinstance(location, Parameter)
    assert isinstance(scale, Parameter)
    data = np.array([0.0, 1.0, 2.0])
    starting_state = {location: np.asarray(0.5), scale: np.asarray(2.0)}
    starting_values = {
        parameter: value.copy() for parameter, value in starting_state.items()
    }

    result = fit_mle(
        model, data, state=starting_state, fixed={location: np.asarray(0.0)}
    )

    assert isinstance(result, FitResult)
    assert result.model is model
    assert result.optimize_result.success
    assert tuple(result.fixed) == (location,)
    assert result.fixed[location] == pytest.approx(0.0)
    assert result.state[location] == pytest.approx(0.0)
    assert result.state[scale] == pytest.approx(np.sqrt(5.0 / 3.0), rel=1e-3)
    assert model.parameters["loc"] is location
    assert model.parameters["scale"] is scale
    assert location.init == 0.0
    assert scale.init == 1.0
    for parameter, value in starting_values.items():
        assert_allclose(starting_state[parameter], value)

    assert negative_log_likelihood(model, data, state=result.state) == pytest.approx(
        -np.sum(stats.norm.logpdf(data, loc=0.0, scale=np.sqrt(5.0 / 3.0)))
    )


def test_fit_result_reports_optimizer_failure() -> None:
    result = fit_mle(Normal(), [0.0, 1.0, 2.0], scipy_args={"options": {"maxiter": 0}})

    assert not result.optimize_result.success


def test_fitting_a_constant_model_has_no_free_state() -> None:
    model = Normal(loc=2.0, scale=3.0)
    data = np.array([1.0, 2.0, 3.0])

    result = fit_mle(model, data)

    assert result.state == {}
    assert result.optimize_result.success
    assert result.optimize_result.x.size == 0
    assert result.optimize_result.fun == pytest.approx(
        -np.sum(stats.norm.logpdf(data, loc=2.0, scale=3.0))
    )


def test_state_supplies_a_start_for_an_uninitialized_parameter() -> None:
    location = Parameter()
    model = Normal(loc=location, scale=1.0)

    with pytest.raises(ValueError, match="uninitialized parameters"):
        fit_mle(model, [1.0, 2.0, 3.0])

    result = fit_mle(model, [1.0, 2.0, 3.0], state={location: 0.0})

    assert result.optimize_result.success
    assert result.state[location] == pytest.approx(2.0, abs=1e-3)
