import numpy as np
import pandas as pd
import pytest

from pykelihood import kernels
from pykelihood.distributions import GEV
from pykelihood.profiler import Profiler


@pytest.fixture(scope="module")
def profiler(dataset):
    fit = GEV().fit(dataset)
    return Profiler(fit, dataset)


@pytest.fixture(scope="module")
def profiler_with_single_profiling_param(dataset):
    fit = GEV().fit(dataset)
    return Profiler(fit, dataset, single_profiling_param="shape")


@pytest.fixture(scope="module")
def profiler_with_fixed_param(dataset):
    fit = GEV().fit(dataset, scale=1.0)
    return Profiler(fit, dataset)


@pytest.fixture(scope="module")
def profiler_with_trend(dataset):
    fit = GEV().fit(
        dataset, loc=kernels.linear(np.linspace(1, len(dataset), len(dataset)))
    )
    return Profiler(fit, dataset)


def test_mle(profiler, dataset):
    mle, profiler_opt = profiler.optimum
    assert profiler_opt == mle.logpdf(dataset).sum()
    # checks whether the maximum likelihood fitted distribution has the same structure as the reference distribution
    assert len(profiler.distribution.flattened_params) == len(mle.flattened_params)
    assert len(profiler.distribution.optimisation_params) == len(
        mle.optimisation_params
    )


def test_mle_with_trend(profiler_with_trend, dataset):
    mle, profiler_opt = profiler_with_trend.optimum
    assert profiler_opt == mle.logpdf(dataset).sum()
    assert len(profiler_with_trend.distribution.flattened_params) == len(
        mle.flattened_params
    )
    assert len(profiler_with_trend.distribution.optimisation_params) == len(
        mle.optimisation_params
    )


def test_mle_with_fixed_param(profiler_with_fixed_param, dataset):
    mle, profiler_opt = profiler_with_fixed_param.optimum
    assert profiler_opt == mle.logpdf(dataset).sum()
    assert len(profiler_with_fixed_param.distribution.flattened_params) == len(
        mle.flattened_params
    )
    assert len(profiler_with_fixed_param.distribution.optimisation_params) == len(
        mle.optimisation_params
    )


def test_profiles(profiler):
    profiles = profiler.profiles
    mle, profiler_opt = profiler.optimum
    # checks that the profiling is made on optimized params and not on fixed ones
    assert len(profiles) == len(mle.optimisation_params)
    for key in profiles:
        # if the likelihood is very concave in one of the parameter, moving slightly away from the MLE can engender a too big deviation from the optimal likelihood value
        if len(profiles[key]):
            # the max likelihood estimate should provide the biggest likelihood for the same set of data and the same assumed distribution structure
            assert pd.Series(profiles[key]["score"] <= profiler_opt).all()
            # a profile is a combination of the parameters of the distribution obtained by fixing one parameter (the one that is being profiled) and
            # fitting the MLE for the sample data and the likelihood value: it should provide a complete view of the fit and therefore contains all
            # of the parameters, even the ones that do not intervene in the optimisation at all
            assert len(profiles[key].columns) == len(mle.flattened_params) + 1
            # the mle for this parameter should be within the bounds found by varying it
            assert (
                profiles[key][key].min()
                <= mle.optimisation_param_dict[key]()
                <= profiles[key][key].max()
            )


def test_profiles_with_single_profiling_param(profiler_with_single_profiling_param):
    single_param_profiles = profiler_with_single_profiling_param.profiles
    assert len(single_param_profiles) == 1
    assert "shape" in single_param_profiles


def test_profiles_with_trend(profiler_with_trend):
    profiles = profiler_with_trend.profiles
    mle, profiler_opt = profiler_with_trend.optimum
    assert len(profiles) == len(mle.optimisation_params)
    for key in profiles:
        if len(profiles[key]):
            assert np.all(profiles[key]["score"] < profiler_opt + 1e-8)
            assert len(profiles[key].columns) == len(mle.flattened_params) + 1
            assert (
                profiles[key][key].min()
                <= mle.optimisation_param_dict[key]()
                <= profiles[key][key].max()
            )


def test_profiles_with_fixed_param(profiler_with_fixed_param):
    profiles = profiler_with_fixed_param.profiles
    mle, profiler_opt = profiler_with_fixed_param.optimum
    assert len(profiles) == len(mle.optimisation_params)
    for key in profiles:
        if len(profiles[key]):
            assert pd.Series(profiles[key]["score"] <= profiler_opt).all()
            assert len(profiles[key].columns) == len(mle.flattened_params) + 1
            assert (
                profiles[key][key].min()
                <= mle.optimisation_param_dict[key]()
                <= profiles[key][key].max()
            )


def test_confidence_interval(profiler):
    mle, _ = profiler.optimum
    CI = profiler.confidence_interval("shape")
    assert CI[0] <= mle.shape.value
    assert mle.shape.value <= CI[1]
