import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from pykelihood.distributions.scipy_wrappers import Normal
from pykelihood.parameters import Parameter
from pykelihood.parametric import ProfilePoint, Profiler, fit_mle
from pykelihood.state import ParameterLayout, PositiveTransform


def test_normal_mean_profile_and_interval_at_zero_mle() -> None:
    location = Parameter(init=0.0)
    model = Normal(loc=location, scale=1.0)
    data = np.array([-2.0, -1.0, 1.0, 2.0])
    fit_result = fit_mle(model, data)
    profiler = Profiler(fit_result, data, confidence=0.95)

    points = profiler.profile(location, [-0.5, 0.0, 0.5])
    assert all(isinstance(point, ProfilePoint) for point in points)
    assert [point.state[location].item() for point in points] == [-0.5, 0.0, 0.5]
    assert all(point.fit_result.model is model for point in points)
    assert all(
        point.log_likelihood <= profiler.max_log_likelihood + 1e-8 for point in points
    )

    interval = profiler.confidence_interval(location)
    expected_half_width = np.sqrt(chi2.ppf(0.95, df=1) / len(data))
    assert_allclose(interval, [-expected_half_width, expected_half_width], atol=2e-3)
    assert location.init == 0.0
    assert model.parameters["loc"] is location


def test_profile_refits_nuisance_and_fixes_shared_parameter_identity() -> None:
    offset = Parameter(init=0.0)
    shared = Parameter(init=1.0, transform=PositiveTransform())
    model = Normal(loc=offset + shared, scale=shared)
    data = np.array([-0.8, -0.2, 0.1, 0.7, 1.1])
    fit_result = fit_mle(model, data)
    profiler = Profiler(fit_result, data)

    layout = ParameterLayout.from_expr(model)
    assert layout.parameters == (offset, shared)
    assert layout.parameter_paths[shared] == (("loc", "right"), ("scale",))

    candidates = (0.6, 1.2)
    points = profiler.profile(shared, candidates)
    for candidate, point in zip(candidates, points):
        assert point.state[shared] == pytest.approx(candidate)
        assert point.fit_result.fixed[shared] == pytest.approx(candidate)
        assert point.fit_result.optimize_result.success
        assert np.isfinite(point.state[offset])
        assert np.isfinite(point.log_likelihood)


def test_positive_scale_interval_brackets_inside_transform_domain() -> None:
    scale = Parameter(init=1.0, transform=PositiveTransform())
    model = Normal(loc=0.0, scale=scale)
    data = np.array([-1.0, 0.0, 1.0])
    fit_result = fit_mle(model, data)
    profiler = Profiler(fit_result, data)

    lower, upper = profiler.confidence_interval(scale, step=2.0)

    assert 0.0 < lower < fit_result.state[scale] < upper


def test_profiler_rejects_fixed_or_non_scalar_parameters() -> None:
    fixed = Parameter(init=0.0)
    fixed_model = Normal(loc=fixed, scale=1.0)
    fixed_fit = fit_mle(fixed_model, [0.0, 1.0], fixed={fixed: 0.0})
    with pytest.raises(ValueError, match="fixed in the original fit"):
        Profiler(fixed_fit, [0.0, 1.0]).profile(fixed, [0.0, 1.0])

    vector = Parameter(init=np.array([0.0, 1.0]))
    vector_model = Normal(loc=vector, scale=1.0)
    vector_fit = fit_mle(vector_model, [0.0, 1.0])
    with pytest.raises(ValueError, match="Only scalar parameters"):
        Profiler(vector_fit, [0.0, 1.0]).profile(vector, [0.0])
