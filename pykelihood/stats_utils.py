from __future__ import annotations

import math
import warnings
from functools import partial
from itertools import count
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from pykelihood.cached_property import cached_property
from pykelihood.distributions import GPD, Distribution, Exponential
from pykelihood.metrics import (
    AIC,
    BIC,
    Brier_score,
    bootstrap,
    crps,
    opposite_log_likelihood,
    qq_l1_distance,
    quantile_score,
)
from pykelihood.parameters import ParametrizedFunction

warnings.filterwarnings("ignore")


class Profiler(object):
    def __init__(
        self,
        distribution: Distribution,
        data: pd.Series,
        score_function: Callable = opposite_log_likelihood,
        name: str = "Standard",
        inference_confidence: float = 0.99,
        single_profiling_param=None,
    ):
        """

        :param distribution: distribution on which the inference is based
        :param data: variable of interest
        :param score_function: function used for optimisation
        :param name: name (optional) of the likelihood if it needs to be compared to other likelihood functions
        :param inference_confidence: wanted confidence for intervals
        :param fit_chi2: whether the results from the likelihood ratio method must be fitted to a chi2
        or a generic chi2 with degree of freedom 1 is used
        :param single_profiling_param: parameter that we want to fix to create the profiles based on likelihood
        """
        self.name = name
        self.distribution = distribution
        self.data = data
        self.score_function = score_function
        self.inference_confidence = inference_confidence
        self.single_profiling_param = single_profiling_param

    @cached_property
    def standard_mle(self):
        estimate = self.distribution.fit(self.data)
        ll = -opposite_log_likelihood(estimate, self.data)
        ll = ll if isinstance(ll, float) else ll[0]
        return (estimate, ll)

    @cached_property
    def optimum(self):
        x0 = self.distribution.optimisation_params
        estimate = self.distribution.fit_instance(
            self.data, score=self.score_function, x0=x0
        )
        func = -self.score_function(estimate, self.data)
        func = func if isinstance(func, float) else func[0]
        return (estimate, func)

    @cached_property
    def profiles(self):
        profiles = {}
        opt, func = self.optimum
        if self.single_profiling_param is not None:
            params = [self.single_profiling_param]
        else:
            params = opt.optimisation_param_dict.keys()
        for name, k in opt.optimisation_param_dict.items():
            if name in params:
                r = k.real
                lb = (
                    r - 1 * (10 ** math.floor(math.log10(np.abs(r))))
                    if name != "shape"
                    else -1.0
                )
                ub = (
                    r + 1 * (10 ** math.floor(math.log10(np.abs(r))))
                    if name != "shape"
                    else 1.0
                )
                range = list(np.linspace(lb, ub, 50))
                profiles[name] = self.test_profile_likelihood(range, name)
        return profiles

    def test_profile_likelihood(self, range_for_param, param):
        opt, func = self.optimum
        profile_ll = []
        params = []
        for x in range_for_param:
            try:
                pl = opt.fit_instance(
                    self.data,
                    score=self.score_function,
                    **{param: x},
                )
                pl_value = -self.score_function(pl, self.data)
                pl_value = pl_value if isinstance(pl_value, float) else pl_value[0]
                if np.isfinite(pl_value):
                    profile_ll.append(pl_value)
                    params.append([p.value for p in pl.flattened_params])
            except:
                pass
        chi2_par = {"df": 1}
        lower_bound = func - chi2.ppf(self.inference_confidence, **chi2_par) / 2
        filtered_params = pd.DataFrame(
            [x + [eval] for x, eval in zip(params, profile_ll) if eval >= lower_bound]
        )
        cols = list(opt.flattened_param_dict.keys()) + ["score"]
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
            columns = list(self.optimum[0].optimisation_param_dict.keys())
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
            return [-np.inf, np.inf]


class DetrendedFluctuationAnalysis(object):
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
        if scale_lim is None:
            lim_inf = 10 ** (math.floor(np.log10(len(data))) - 1)
            lim_sup = min(
                10 ** (math.ceil(np.log10(len(data)))), len(phi)
            )  # assuming all observations are equally splitted
            scale_lim = [lim_inf, lim_sup]
        if scale_step is None:
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


def pettitt_test(data: Union[np.array, pd.DataFrame, pd.Series]):
    """
    Pettitt's non-parametric test for change-point detection.
    Given an input signal, it reports the likely position of a single switch point along with
    the significance probability for location K, approximated for p <= 0.05.
    """
    T = len(data)
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        X = np.array(data).reshape((len(data), 1))
    else:
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


def threshold_selection_gpd_NorthorpColeman(
    data: Union[pd.Series, np.ndarray],
    thresholds: Union[Sequence, np.ndarray],
    plot=False,
):
    """
    Method based on a multiple threshold penultimate model,
    introduced by Northorp and Coleman in 2013 for threshold selection in extreme value analysis.
    Returns: table with likelihood computed using hypothesis of constant parameters h0, ll with h1, p_value of test, # obs
    and figure to plot as an option
    """
    if isinstance(data, pd.Series):
        data = data.rename("realized")
    elif isinstance(data, np.ndarray):
        data = pd.Series(data, name="realized")
    else:
        return TypeError("Observations should be in array or pandas series format.")
    if isinstance(thresholds, Sequence):
        thresholds = np.array(thresholds)

    fits = {}
    nll_ref = {}
    for u in thresholds:
        d = data[data > u]
        fits[u] = GPD.fit(d, loc=u)
        nll_ref[u] = opposite_log_likelihood(fits[u], d)

    def negated_ll(x: np.array, ref_threshold: float):
        tol = 1e-10
        sigma_init = x[0]
        if sigma_init <= tol:
            return 10 ** 10
        xi_init = x[1:]
        # It could be interesting to consider this parameter stability condition
        if len(xi_init[np.abs(xi_init) >= 1.0]):
            return 10 ** 10
        thresh = [u for u in thresholds if u >= ref_threshold]
        thresh_diff = pd.Series(
            np.concatenate([np.diff(thresh), [np.nan]]), index=thresh, name="w"
        )
        xi = pd.Series(xi_init, index=thresh, name="xi")
        sigma = pd.Series(
            sigma_init
            + np.cumsum(np.concatenate([[0], xi.iloc[:-1] * thresh_diff.iloc[:-1]])),
            index=thresh,
            name="sigma",
        )
        params_and_conditions = (
            pd.concat([sigma, xi, thresh_diff], axis=1)
            .assign(
                positivity_p_condition=lambda x: (1 + (x["xi"] * x["w"]) / x["sigma"])
                .shift(1)
                .fillna(1.0)
            )
            .assign(
                logp=lambda x: np.cumsum(
                    (1 / x["xi"]) * np.log(1 + x["xi"] * x["w"] / x["sigma"])
                )
                .shift(1)
                .fillna(0.0)
            )
            .reset_index()
            .rename(columns={"index": "lb"})
        )
        if np.any(params_and_conditions["positivity_p_condition"] <= tol):
            return 1 / tol
        thresh_for_pairs = np.concatenate([thresh, [data.max() + 1]])
        y_cut = (
            pd.concat(
                [
                    data,
                    pd.Series(
                        np.array(
                            pd.cut(
                                data, thresh_for_pairs, right=True, include_lowest=False
                            ).apply(lambda x: x.left)
                        ),
                        name="lb",
                        index=data.index,
                    ),
                ],
                axis=1,
            )
            .dropna()
            .merge(
                params_and_conditions.drop(columns=["w", "positivity_p_condition"]),
                on="lb",
                how="left",
            )
        )
        if (
            1 + y_cut["xi"] * (y_cut["realized"] - y_cut["lb"]) / y_cut["sigma"] <= tol
        ).any():
            return 1 / tol
        y_cut = y_cut.assign(
            nlogl=lambda x: x["logp"]
            + np.log(x["sigma"])
            + (1 + 1 / x["xi"])
            * np.log(1 + x["xi"] * (x["realized"] - x["lb"]) / x["sigma"])
        )
        logl_per_interval = y_cut.groupby("lb").agg({"nlogl": "sum"})
        return logl_per_interval[np.isfinite(logl_per_interval)].sum()[0]

    par = {}
    results_dic = {}
    u_test = [thresholds[-1] + i for i in [5, 10, 20]]
    for u in thresholds:
        sigma_init, xi_1 = fits[u].optimisation_params
        xi_init = np.array([xi_1] * len(thresholds[thresholds >= u]))
        to_minimize = partial(negated_ll, ref_threshold=u)
        x0 = np.concatenate([[sigma_init], xi_init])
        params = minimize(
            to_minimize,
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 10000, "fatol": 0.05},
        )
        print(f"Threshold {u}: ", params.message)
        mle = params.x
        nll = params.fun
        nll_h0 = to_minimize(x0)
        par[u] = mle
        delta = 2 * (nll_h0 - nll)
        df = len(thresholds[thresholds >= u]) - 1
        p_value = chi2.sf(delta, df=df)
        aic = AIC(fits[u], data[data > u])
        bic = BIC(fits[u], data[data > u])
        crpss = crps(fits[u], data[data > u])
        results_dic[u] = {
            "nobs": len(data[data > u]),
            "nll_h0": nll_h0,
            "nll_h1": nll,
            "pvalue": p_value,
            "aic": aic,
            "bic": bic,
            "crps": crpss,
        }
        for t in u_test:
            results_dic[u][f"bs_{int(t)}"] = Brier_score(
                fits[u], data[data > u], threshold=t
            )
        for q in [0.9, 0.95]:
            results_dic[u][f"qs_{int(q * 100)}"] = quantile_score(
                fits[u], np.quantile(data[data > u], q), quantile=q
            )
    results = pd.DataFrame.from_dict(results_dic, orient="index")
    if not plot:
        return results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(results.index, results["pvalue"], color="navy", label="p-value")
    ax.scatter(results.index, results["pvalue"], marker="x", s=8, color="navy")
    ax.legend(loc="upper center", title="LR test")
    ax2 = ax.twinx()
    if results["crps"].any():
        results = results.assign(
            rescaled_crps=lambda x: x["crps"] * (x["qs_90"].mean() / x["crps"].mean())
        )
        ax2.plot(
            results.dropna(subset=["pvalue"])["rescaled_crps"],
            c="royalblue",
            label="CRPS",
        )
    for f, c in zip([90, 95], ["salmon", "goldenrod"]):
        ax2.plot(
            results.dropna(subset=["pvalue"])[f"qs_{f}"],
            c=c,
            label=r"${}\%$ Quantile Score".format(f),
        )
    ax2.set_ylabel("GoF scores")
    ax2.legend(title="GoF scores")
    ax.set_title("Threshold Selection plot based on LR test")
    ax.set_xlabel("threshold")
    ax.set_ylabel("p-value")
    fig.show()
    return results, fig


def threshold_selection_GoF(
    data: Union[pd.Series, np.ndarray],
    min_threshold: float,
    max_threshold: float,
    metric=qq_l1_distance,
    plot=False,
):
    """
    Threshold selection method based on goodness of fit maximisation.
    Introduced by Varty, Z., Tawn, J. A., Atkinson, P. M., & Bierman, S. (2021).
    Inference for extreme earthquake magnitudes accounting for a time-varying measurement process. arXiv preprint arXiv:2102.00884.
    :return: table with results and plot (if selected)
    """
    from copy import copy

    if isinstance(data, pd.DataFrame):
        data_series = data[data_column]
        if data_column is None:
            raise AttributeError(
                "The data column is needed to perform the threshold selection on a DataFrame."
            )
    else:
        data_series = data

    if GPD_instance is None:
        GPD_instance = GPD()

    def update_covariates(GPD_instance, threshold):
        count = 0
        temp_gpd = copy(GPD_instance)
        for param in GPD_instance.param_dict:
            if isinstance(GPD_instance.__getattr__(param), ParametrizedFunction):
                # tracking number of functional parameters
                count += 1
                parametrized_param = getattr(GPD_instance, param)
                covariate = parametrized_param.x
                new_covariate = data[data[data_column] > threshold][covariate.name]
                temp_gpd = temp_gpd.with_params(
                    **{param: GPD_instance.loc.with_covariate(new_covariate)}
                )
        return temp_gpd, count

    def to_minimize(x):
        threshold = x[0]
        unit_exp = Exponential()
        if bootstrap_method is None:
            print("Performing threshold selection without bootstrap...")
            new_data = data_series[data_series > threshold]
            temp_gpd, count = update_covariates(GPD_instance, threshold)
            gpd_fit = temp_gpd.fit_instance(new_data, x0=GPD_instance.flattened_params)
            if count > 0:
                return metric(
                    distribution=unit_exp,
                    data=unit_exp.inverse_cdf(gpd_fit.cdf(new_data)),
                )
            else:
                return metric(distribution=gpd_fit, data=new_data)
        else:
            bootstrap_func = partial(bootstrap_method, threshold=threshold)
            new_metric = bootstrap(metric, bootstrap_func)
            return new_metric(unit_exp, data)

    threshold_sequence = np.linspace(min_threshold, max_threshold, 30)
    func_eval = np.array([to_minimize([t]) for t in threshold_sequence])
    nans_inf = np.isnan(func_eval) | ~np.isfinite(func_eval)
    func_eval = func_eval[~nans_inf]
    threshold_sequence = threshold_sequence[~nans_inf]

    optimal_thresh = threshold_sequence[np.where(func_eval == np.min(func_eval))[0]][0]
    func = np.min(func_eval)
    res = [optimal_thresh, func]

    if not plot:
        return res
    import matplotlib.pyplot as plt

    data_to_fit = data_series[data_series > optimal_thresh]
    optimal_gpd, count = update_covariates(GPD_instance, optimal_thresh)
    gpd_fit = optimal_gpd.fit_instance(data_to_fit)
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax = []
    for i in range(1):
        for j in range(2):
            axe = fig.add_subplot(gs[i, j])
            ax.append(axe)
    plt.subplots_adjust(wspace=0.2)
    ax0, ax1 = ax
    ax0.plot(threshold_sequence, func_eval, color="navy")
    ax0.scatter(threshold_sequence, func_eval, marker="x", s=8, color="navy")
    ax0.vlines(
        optimal_thresh,
        func,
        np.max(func_eval),
        color="royalblue",
        label="Optimal threshold",
    )
    ax0.set_title("Threshold Selection plot based on GoF")
    ax0.set_xlabel("threshold")
    ax0.set_ylabel(metric.__name__.replace("_", " "))

    if metric.__name__.startswith("qq"):
        if count > 0 or bootstrap_method is not None:
            from pykelihood.visualisation.utils import qq_plot_exponential_scale

            qq_plot_exponential_scale(gpd_fit, data_to_fit, ax=ax1)
        else:
            from pykelihood.visualisation.utils import qq_plot

            qq_plot(gpd_fit, data_to_fit, ax=ax1)
    elif metric.__name__.startswith("pp"):
        from pykelihood.visualisation.utils import pp_plot

        pp_plot(gpd_fit, data_to_fit, ax=ax1)
    fig.show()
    return res, fig
