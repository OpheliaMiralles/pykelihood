import numpy as np
import pandas as pd
import pytest

from pykelihood import kernels
from pykelihood.distributions import Distribution, GEV
from pykelihood.stats_utils import Likelihood


@pytest.fixture(scope="module")
def likelihood(dataset):
    fit = GEV.fit(dataset)
    return Likelihood(fit, dataset)


@pytest.fixture(scope="module")
def likelihood_with_single_profiling_param(dataset):
    fit = GEV.fit(dataset)
    return Likelihood(fit, dataset, single_profiling_param='shape')


@pytest.fixture(scope="module")
def likelihood_with_fixed_param(dataset):
    fit = GEV.fit(dataset, scale=1.)
    return Likelihood(fit, dataset)


@pytest.fixture(scope="module")
def likelihood_with_trend(dataset):
    fit = GEV.fit(dataset, loc=kernels.linear(np.linspace(1, len(dataset), len(dataset))))
    return Likelihood(fit, dataset)


def test_mle(likelihood, dataset):
    mle, likelihood_opt = likelihood.mle
    assert isinstance(mle, Distribution)
    assert likelihood_opt == mle.logpdf(dataset).sum()
    # checks whether the maximum likelihood fitted distribution has the same structure as the reference distribution
    assert len(likelihood.distribution.flattened_params) == len(mle.flattened_params)
    assert len(likelihood.distribution.optimisation_params) == len(mle.optimisation_params)


def test_mle_with_trend(likelihood_with_trend, dataset):
    mle, likelihood_opt = likelihood_with_trend.mle
    assert isinstance(mle, Distribution)
    assert likelihood_opt == mle.logpdf(dataset).sum()
    assert len(likelihood_with_trend.distribution.flattened_params) == len(mle.flattened_params)
    assert len(likelihood_with_trend.distribution.optimisation_params) == len(mle.optimisation_params)


def test_mle_with_fixed_param(likelihood_with_fixed_param, dataset):
    mle, likelihood_opt = likelihood_with_fixed_param.mle
    assert isinstance(mle, Distribution)
    assert likelihood_opt == mle.logpdf(dataset).sum()
    assert len(likelihood_with_fixed_param.distribution.flattened_params) == len(mle.flattened_params)
    assert len(likelihood_with_fixed_param.distribution.optimisation_params) == len(mle.optimisation_params)


def test_profiles(likelihood):
    profiles = likelihood.profiles
    mle, likelihood_opt = likelihood.mle
    # checks that the profiling is made on optimized params and not on fixed ones
    assert len(profiles) == len(mle.optimisation_params)
    for key in profiles:
        # if the likelihood is very concave in one of the parameter, moving slightly away from the MLE can engender a too big deviation from the optimal likelihood value
        if len(profiles[key]):
            # the max likelihood estimate should provide the biggest likelihood for the same set of data and the same assumed distribution structure
            assert pd.Series((profiles[key]['likelihood'] <= likelihood_opt)).all()
            # a profile is a combination of the parameters of the distribution obtained by fixing one parameter (the one that is being profiled) and
            # fitting the MLE for the sample data and the likelihood value: it should provide a complete view of the fit and therefore contains all
            # of the parameters, even the ones that do not intervene in the optimisation at all
            assert len(profiles[key].columns) == len(mle.flattened_params) + 1
            # the mle for this parameter should be within the bounds found by varying it
            assert profiles[key][key].min() <= mle.optimisation_param_dict[key] <= profiles[key][key].max()


def test_profiles_with_single_profiling_param(likelihood_with_single_profiling_param):
    single_param_profiles = likelihood_with_single_profiling_param.profiles
    assert len(single_param_profiles) == 1
    assert 'shape' in single_param_profiles


def test_profiles_with_trend(likelihood_with_trend):
    profiles = likelihood_with_trend.profiles
    mle, likelihood_opt = likelihood_with_trend.mle
    assert len(profiles) == len(mle.optimisation_params)
    for key in profiles:
        if len(profiles[key]):
            assert pd.Series((profiles[key]['likelihood'] <= likelihood_opt)).all()
            assert len(profiles[key].columns) == len(mle.flattened_params) + 1
            assert profiles[key][key].min() <= mle.optimisation_param_dict[key] <= profiles[key][key].max()


def test_profiles_with_fixed_param(likelihood_with_fixed_param):
    profiles = likelihood_with_fixed_param.profiles
    mle, likelihood_opt = likelihood_with_fixed_param.mle
    assert len(profiles) == len(mle.optimisation_params)
    for key in profiles:
        if len(profiles[key]):
            assert pd.Series((profiles[key]['likelihood'] <= likelihood_opt)).all()
            assert len(profiles[key].columns) == len(mle.flattened_params) + 1
            assert profiles[key][key].min() <= mle.optimisation_param_dict[key] <= profiles[key][key].max()


def test_confidence_interval(likelihood, likelihood_with_single_profiling_param):
    return_periods = np.linspace(20, 200, 5)
    mle, likelihood_opt = likelihood.mle
    assert likelihood.inference_confidence == 0.99
    for return_period in return_periods:
        def metric(distribution):
            return distribution.isf(1 / return_period)

        estimated_level = metric(mle)
        CI = likelihood.confidence_interval(metric)
        single_param_CI = likelihood_with_single_profiling_param.confidence_interval(metric)
        assert CI[0] <= estimated_level <= CI[1]
        assert CI[0] <= single_param_CI[0]
        # profiling according to only one parameter gives less wide and less reliable confidence intervals
        assert CI[1] >= single_param_CI[1]
