from __future__ import annotations

import math
import warnings
from itertools import count
from typing import TYPE_CHECKING, Callable, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from pykelihood.cached_property import cached_property

if TYPE_CHECKING:
    from pykelihood.distributions import Distribution

warnings.filterwarnings("ignore")


class ConditioningMethod(object):
    @staticmethod
    def no_conditioning(data: pd.Series, distribution: Distribution):
        return 0.0

    @staticmethod
    def excluding_last_obs_rule(data: pd.Series, distribution: Distribution):
        return distribution.logpdf(data.iloc[-1])

    @staticmethod
    def partial_conditioning_rule_stopped_obs(
        data: pd.Series, distribution: Distribution, threshold: Sequence = None
    ):
        return distribution.logsf(threshold[-1])

    @staticmethod
    def full_conditioning_rule_stopped_obs(
        data: pd.Series, distribution: Distribution, threshold: Sequence = None
    ):
        return distribution.logsf(threshold[-1]) + np.sum(
            distribution.logcdf(threshold[:-1])
        )


class Likelihood(object):
    def __init__(
        self,
        distribution: Distribution,
        data: pd.Series,
        conditioning_method: Callable = ConditioningMethod.no_conditioning,
        name: str = "Standard",
        inference_confidence: float = 0.99,
        fit_chi2: bool = False,
        single_profiling_param=None,
        compute_mle=True,
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
        self.compute_mle = compute_mle

    @cached_property
    def standard_mle(self):
        estimate = self.distribution.fit(self.data)
        ll = estimate.log_likelihood(self.data)
        ll = ll if isinstance(ll, float) else ll[0]
        return (estimate, ll)

    @cached_property
    def mle(self):
        if self.compute_mle:
            x0 = self.distribution.optimisation_params
            estimate = self.distribution.fit_instance(
                self.data, penalty=self.conditioning_method, x0=x0
            )
        else:
            estimate = self.distribution
        ll_xi0 = estimate.log_likelihood(self.data, penalty=self.conditioning_method)
        ll_xi0 = ll_xi0 if isinstance(ll_xi0, float) else ll_xi0[0]
        return (estimate, ll_xi0)

    @cached_property
    def AIC(self):
        mle_aic = -2 * self.mle[1] + 2 * len(self.mle[0].optimisation_params)
        std_mle_aic = -2 * self.standard_mle[1] + 2 * len(
            self.standard_mle[0].optimisation_params
        )
        return {"AIC MLE": mle_aic, "AIC Standard MLE Fit": std_mle_aic}

    def Deviance(self):
        mle_deviance = -2 * self.mle[1]
        std_mle_deviance = -2 * self.standard_mle[1]
        return {
            "Deviance MLE": mle_deviance,
            "AIC Standard MLE Deviance": std_mle_deviance,
        }

    @cached_property
    def profiles(self):
        profiles = {}
        mle, ll_xi0 = self.mle
        if self.single_profiling_param is not None:
            params = [self.single_profiling_param]
        else:
            params = mle.optimisation_param_dict.keys()
        for name, k in mle.optimisation_param_dict.items():
            if name in params:
                r = k.real
                lb = r - 5 * (10 ** math.floor(math.log10(np.abs(r))))
                ub = r + 5 * (10 ** math.floor(math.log10(np.abs(r))))
                range = list(np.linspace(lb, ub, 50))
                profiles[name] = self.test_profile_likelihood(range, name)
        return profiles

    def test_profile_likelihood(self, range_for_param, param):
        mle, ll_xi0 = self.mle
        profile_ll = []
        params = []
        for x in range_for_param:
            try:
                pl = mle.fit_instance(
                    self.data,
                    penalty=self.conditioning_method,
                    fixed_params={param: x},
                )
                pl_value = pl.log_likelihood(
                    self.data, penalty=self.conditioning_method
                )
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
        filtered_params = pd.DataFrame(
            [x + [ll] for x, ll in zip(params, profile_ll) if ll >= lower_bound]
        )
        cols = list(mle.flattened_param_dict.keys()) + ["likelihood"]
        filtered_params = filtered_params.rename(columns=dict(zip(count(), cols)))
        return filtered_params

    def confidence_interval(self, metric: Callable[[Distribution], float]):
        """

        :param metric: function depending on the distribution: it can be one of the parameter (ex: lambda x: x.shape() for a parameter called "shape"),
        or a metric relevant to the field of study (ex: the 100-years return level for extreme value analysis by setting lambda x: x.isf(1/100))...
        :return: bounds based on parameter profiles for this metric
        """
        estimates = []
        profiles = self.profiles
        if self.single_profiling_param is not None:
            params = [self.single_profiling_param]
        else:
            params = profiles.keys()
        for param in params:
            columns = list(self.mle[0].optimisation_param_dict.keys())
            result = profiles[param].apply(
                lambda row: metric(
                    self.distribution.with_params({k: row[k] for k in columns}.values())
                ),
                axis=1,
            )
            estimates.extend(list(result.values))
        if len(estimates):
            return [np.min(estimates), np.max(estimates)]
        else:
            return [None, None]


class DetrentedFluctuationAnalysis(object):
    def __init__(
        self,
        data: pd.DataFrame,
        scale_lim: Sequence[int] = None,
        scale_step: float = None,
    ):
        """

        :param data: pandas Dataframe, if it contains a column for the day and month, the profiles are normalized
        according to the mean for each calendar day averaged over years.
        :param scale_lim: limits for window sizes
        :param scale_step: steps for window sizes
        """
        if not ("month" in data.columns and "day" in data.columns):
            print("Will use the total average to normalize the data...")
            mean = data["data"].mean()
            std = data["data"].std()
            data = data.assign(mean=mean).assign(std=std)
        else:
            mean = (
                data.groupby(["month", "day"])
                .agg({"data": "mean"})["data"]
                .rename("mean")
                .reset_index()
            )
            std = (
                data.groupby(["month", "day"])
                .agg({"data": "std"})["data"]
                .rename("std")
                .reset_index()
            )
            data = data.merge(mean, on=["month", "day"], how="left").merge(
                std, on=["month", "day"], how="left"
            )
        phi = (data["data"] - data["mean"]) / data["std"]
        phi = (
            phi.dropna()
        )  # cases where there is only one value for a given day / irrelevant for DFA
        self.y = np.cumsum(np.array(phi))
        if scale_lim == None:
            lim_inf = 10 ** (math.floor(np.log10(len(data))) - 1)
            lim_sup = min(
                10 ** (math.ceil(np.log10(len(data)))), len(phi)
            )  # assuming all observations are equally splitted
            scale_lim = [lim_inf, lim_sup]
        if scale_step == None:
            scale_step = 10 ** (math.floor(np.log10(len(data)))) / 2
        self.scale_lim = scale_lim
        self.scale_step = scale_step

    @staticmethod
    def calc_rms(x: np.array, scale: int, polynomial_order: int):
        """
        windowed Root Mean Square (RMS) with polynomial detrending.
        Args:
        -----
          *x* : numpy.array
            one dimensional data vector
          *scale* : int
            length of the window in which RMS will be calculaed
        Returns:
        --------
          *rms* : numpy.array
            RMS data in each window with length len(x)//scale
        """
        # making an array with data divided in windows
        shape = (x.shape[0] // scale, scale)
        X = np.lib.stride_tricks.as_strided(x, shape=shape)
        # vector of x-axis points to regression
        scale_ax = np.arange(scale)
        rms = np.zeros(X.shape[0])
        for e, xcut in enumerate(X):
            coeff = np.polyfit(scale_ax, xcut, deg=polynomial_order)
            xfit = np.polyval(coeff, scale_ax)
            # detrending and computing RMS of each window
            rms[e] = np.mean((xcut - xfit) ** 2)
        return rms

    @staticmethod
    def trend_type(alpha: float):
        if round(alpha, 1) < 1:
            if round(alpha, 1) < 0.5:
                return "Anti-correlated"
            elif round(alpha, 1) == 0.5:
                return "Uncorrelated, white noise"
            elif round(alpha, 1) > 0.5:
                return "Correlated"
        elif round(alpha, 1) == 1:
            return "Noise, pink noise"
        elif round(alpha, 1) > 1:
            if round(alpha, 1) < 1.5:
                return "Non-stationary, unbounded"
            else:
                return "Brownian Noise"

    def __call__(
        self, polynomial_order: int, show=False, ax=None, supplement_title="", color="r"
    ):
        """
        Detrended Fluctuation Analysis - measures power law scaling coefficient
        of the given signal *x*.
        More details about the algorithm you can find e.g. here:
        Kropp, Jürgen, & Schellnhuber, Hans-Joachim. 2010. Case Studies. Chap. 8-11, pages 167–244 of : In extremis :
        disruptive events and trends in climate and hydrology. Springer Science & Business Media.
        """

        y = self.y
        scales = (
            np.arange(self.scale_lim[0], self.scale_lim[1], self.scale_step)
        ).astype(np.int)
        fluct = np.zeros(len(scales))
        # computing RMS for each window
        for e, sc in enumerate(scales):
            fluct[e] = np.sqrt(
                np.mean(self.calc_rms(y, sc, polynomial_order=polynomial_order))
            )
        # as this stage, F^2(s) should be something of the form s^h(2); taking the log should give a linear form of coefficient h(2)
        coeff = np.polyfit(np.log(scales), np.log(fluct), 1)
        # numpy polyfit returns the highest power first
        if show:
            import matplotlib

            matplotlib.rcParams["text.usetex"] = True
            ax = ax or matplotlib.pyplot.gca()
            default_title = "Detrended Fluctuation Analysis"
            title = (
                default_title
                if supplement_title == ""
                else f"{default_title} {supplement_title}"
            )
            fluctfit = np.exp(np.polyval(coeff, np.log(scales)))
            ax.loglog(scales, fluct, "o", color=color, alpha=0.6)
            ax.loglog(
                scales,
                fluctfit,
                color=color,
                alpha=0.6,
                label=r"DFA-{}, {}: $\alpha$={}".format(
                    polynomial_order, self.trend_type(coeff[0]), round(coeff[0], 2)
                ),
            )
            ax.set_title(title)
            ax.set_xlabel(r"$\log_{10}$(time window)")
            ax.set_ylabel(r"$\log_{10}$F(t)")
            ax.legend(loc="lower right", fontsize="small")
        return scales, fluct, coeff[0]


def pettitt(data: Union[np.array, pd.DataFrame, pd.Series]):
    """
    Pettitt's non-parametric test for change-point detection.
    Given an input signal, it reports the likely position of a single switch point along with
    the significance probability for location K, approximated for p <= 0.05.
    """
    T = len(data)
    X = data.reshape((len(data), 1))
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
