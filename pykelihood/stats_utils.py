from __future__ import annotations

import math
import warnings
from itertools import count
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
from rpy2.robjects import FloatVector
from scipy.stats import chi2

from pykelihood.cached_property import cached_property

if TYPE_CHECKING:
    from pykelihood.distributions import Distribution

warnings.filterwarnings('ignore')

class ConditioningMethod(object):
    @staticmethod
    def no_conditioning(data: pd.Series,
                        distribution: Distribution):
        if hasattr(data, "rclass"):
            return FloatVector([0.])
        else:
            return 0.

    @staticmethod
    def excluding_last_obs_rule(data: pd.Series,
                                distribution: Distribution):
        if hasattr(data, "rclass"):
            return FloatVector(data[-1])
        else:
            return distribution.logpdf(data.iloc[-1])

    @staticmethod
    def partial_conditioning_rule_stopped_obs(data: pd.Series, distribution: Distribution,
                                              threshold: Sequence = None):
        return distribution.logsf(threshold[-1])

    @staticmethod
    def full_conditioning_rule_stopped_obs(data: pd.Series, distribution: Distribution,
                                           threshold: Sequence = None):
        return distribution.logsf(threshold[-1]) + np.sum(distribution.logcdf(threshold[:-1]))


class Likelihood(object):
    def __init__(self, distribution: Distribution,
                 data: pd.Series,
                 conditioning_method: Callable = ConditioningMethod.no_conditioning,
                 name: str = "Standard",
                 inference_confidence: float = 0.99,
                 fit_chi2: bool = False,
                 single_profiling_param = None
                 ):
        """

        :param distribution: distribution on which the inference is based
        :param data: variable of interest
        :param conditioning_method: penalisation term for the likelihood
        :param name: name (optional) of the likelihood if it needs to be compared to other likelihood functions
        :param inference_confidence: wanted confidence for intervals
        :param fit_chi2: whether the results from the likelihood ratio method must be fitted to a chi2
        or a generic chi2 with degree of freedom 1 is used
        :param single_profiling_param: parameter that we want to fix to create the profiles based on likelihood
        """
        self.name = name
        self.distribution = distribution
        self.data = data
        self.conditioning_method = conditioning_method
        self.inference_confidence = inference_confidence
        self.fit_chi2 = fit_chi2
        self.single_profiling_param = single_profiling_param

    @cached_property
    def standard_mle(self):
        estimate = self.distribution.fit(self.data)
        ll = estimate.log_likelihood(self.data)
        ll = ll if isinstance(ll, float) else ll[0]
        return (estimate, ll)

    @cached_property
    def mle(self):
        x0 = self.distribution.optimisation_params
        estimate = self.distribution.profile_likelihood(self.data,
                                         conditioning_method=self.conditioning_method,
                                         x0=x0)
        ll_xi0 = estimate.log_likelihood(self.data,
                                                  conditioning_method=self.conditioning_method)
        ll_xi0 = ll_xi0 if isinstance(ll_xi0, float) else ll_xi0[0]
        return (estimate, ll_xi0)

    @cached_property
    def AIC(self):
        mle_aic = -2*self.mle[1]+2*len(self.mle[0].optimisation_params)
        std_mle_aic = -2*self.standard_mle[1]+2*len(self.standard_mle[0].optimisation_params)
        return {"AIC MLE": mle_aic, "AIC Standard MLE Fit": std_mle_aic}

    def Deviance(self):
        mle_deviance = -2*self.mle[1]
        std_mle_deviance = -2*self.standard_mle[1]
        return {"Deviance MLE": mle_deviance, "AIC Standard MLE Deviance": std_mle_deviance}

    @cached_property
    def profiles(self):
        profiles = {}
        mle, ll_xi0 = self.mle
        if self.single_profiling_param is not None:
            params = [self.single_profiling_param]
        else:
            params = mle.optimisation_param_dict.keys()
        if hasattr(self.distribution, "fast_profile_likelihood")\
                and len(mle.optimisation_params) == len(mle.params):
            # we cannot use the profiling from R with a distribution whose parameters
            # are fitted to regression kernels, as they implemented a standard fit with constant
            # parameters.
            try:
                pre_profiled_params = self.distribution.fast_profile_likelihood(self.data, conf=self.inference_confidence)
                if self.conditioning_method == ConditioningMethod.excluding_last_obs_rule:
                    pre_profiled_params = self.distribution.fast_profile_likelihood(self.data.iloc[:-1], conf=self.inference_confidence)
                if self.conditioning_method in [ConditioningMethod.no_conditioning,
                                                ConditioningMethod.excluding_last_obs_rule]:
                    for name in params:
                        columns = list(pre_profiled_params[name].columns)
                        likelihoods = pre_profiled_params[name] \
                            .apply(lambda row: self.distribution.with_params([row[k] for k in columns])\
                                   .log_likelihood(self.data.iloc[:-1]), axis=1)
                        pre_profiled_params[name] = pre_profiled_params[name].assign(likelihood=likelihoods)

                    return pre_profiled_params
                for name in params:
                    min_std = np.min(pre_profiled_params[name][name])
                    max_std = np.max(pre_profiled_params[name][name])
                    lb = min_std - 5 * (10 ** math.floor(math.log10(np.abs(min_std))))
                    ub = max_std + 5 * (10 ** math.floor(math.log10(np.abs(max_std))))
                    range = list(np.linspace(lb, min_std, 5)) \
                            + list(pre_profiled_params[name][name].values) \
                            + list(np.linspace(max_std, ub, 5))
                    profiles[name] = self.test_profile_likelihood(range, name)
                return profiles
            except:
                pass
        for name, k in mle.optimisation_param_dict.items():
            if name in params:
                lb = k - 0.1 - 5 * (10 ** math.floor(math.log10(np.abs(k))))
                ub = k + 0.1 + 5 * (10 ** math.floor(math.log10(np.abs(k))))
                range = list(np.linspace(lb, ub, 20))
                profiles[name] = self.test_profile_likelihood(range, name)
        return profiles

    def test_profile_likelihood(self, range_for_param, param):
        mle, ll_xi0 = self.mle
        profile_ll = []
        params = []
        for x in range_for_param:
            try:
                pl = mle.profile_likelihood(self.data,
                                            conditioning_method=self.conditioning_method,
                                            fixed_params={param: x})
                pl_value = pl.log_likelihood(
                    self.data,
                    conditioning_method=self.conditioning_method)
                pl_value = pl_value if isinstance(pl_value, float) else pl_value[0]
                if np.isfinite(pl_value):
                    profile_ll.append(pl_value)
                    params.append(list(pl.flattened_params))
            except:
                pass
        delta = [2 * (ll_xi0 - ll) for ll in profile_ll if np.isfinite(ll)]
        if self.fit_chi2:
            df, loc, scale = chi2.fit(delta)
            chi2_par = {"df": df, "loc": loc, "scale": scale}
        else:
            chi2_par = {"df": 1}
        lower_bound = ll_xi0 - chi2.ppf(self.inference_confidence, **chi2_par) / 2
        filtered_params = pd.DataFrame([x + [ll] for x, ll in zip(params, profile_ll)
                                        if ll >= lower_bound])
        cols = list(mle.flattened_param_dict.keys()) + ["likelihood"]
        filtered_params = filtered_params.rename(columns=dict(zip(count(), cols)))
        return filtered_params

    def return_level(self, return_period):
        mle, ll_xi0 = self.mle
        return_level = mle.isf(1 / return_period)
        return return_level

    def return_level_confidence_interval(self, return_period):
        rle = []
        profiles = self.profiles
        if self.single_profiling_param is not None:
            params = [self.single_profiling_param]
        else:
            params = profiles.keys()
        for param in params:
            columns = list(profiles.keys())
            return_levels = profiles[param] \
                .apply(
                lambda row: self.distribution.with_params({k: row[k] for k in columns}.values()).isf(1 / return_period),
                axis=1)
            rle.extend(list(return_levels.values))
            if len(rle):
                return [np.min(rle), np.max(rle)]
            else:
                return [None, None]


def pettitt(signal):
    """
    Pure-Python implementation of Pettitt's non-parametric test for change-point detection.
    Given an input signal, it reports the likely position of a single changepoint along with
    the significance probability for location K, approximated for p <= 0.05.
    """
    T = len(signal)
    X = signal.reshape((len(signal), 1))
    vector_of_ones = np.ones([1, len(X)])
    matrix_col_X = np.matmul(X, vector_of_ones)
    matrix_lines_X = matrix_col_X.T
    diff = matrix_lines_X - matrix_col_X
    diff_sign = np.sign(diff)
    U_initial = diff_sign[0, 1:].sum()
    sum_of_each_line = diff_sign[1:].sum(axis=1)
    cs = sum_of_each_line.cumsum()
    U = U_initial + cs
    U = list(U)
    U.insert(0, U_initial)
    loc = np.argmax(np.abs(U))
    K = np.max(np.abs(U))
    p = np.exp(-3 * K ** 2 / (T ** 3 + T ** 2))
    return (loc, p)
