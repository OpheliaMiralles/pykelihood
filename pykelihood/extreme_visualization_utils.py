import math
import warnings
from typing import List, Union

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.rcParams['text.usetex'] = True

from pykelihood import kernels
from pykelihood.distributions import Exponential, MixtureExponentialModel
from pykelihood.stats_utils import Likelihood
try:
    from hawkeslib import PoissonProcess as PP, UnivariateExpHawkesProcess as UEHP
except ImportError:
    PP = UEHP = None
from pykelihood.distributions import Distribution
from pykelihood.parameters import Parameter

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


def GPDQQPlot(data: pd.DataFrame, gpd_fit: Distribution, path_to_figure: str,
              threshold: Union[List, float, int, str] = ''):
    empirical_cdf = pd.Series(data).quantile(np.linspace(0.01, 0.99, len(data)))
    n = len(data)
    quantiles_exp = gpd_fit.inverse_cdf(empirical_cdf.index)
    text_title = ''
    if type(gpd_fit.loc()) is not Parameter:
        loc = {r'$\mu_{}$'.format(a): round(gpd_fit.loc.param_dict[a], 2) for a in gpd_fit.loc.param_dict}
        depending_on = list([g.name for g in gpd_fit.loc.f.args])
        for k in loc:
            text_title += f'{k} = {loc[k]}, '
        text_title += f'depending on variables(s) '
        for l in depending_on:
            l = l.replace('_', ' ')
            text_title += r'\it{' + l + r'}' + ', '
    else:
        loc = round(gpd_fit.loc(), 2)
        text_title += r'$\mu$=' + str(loc) + ', '
    text_title += '\n'
    if type(gpd_fit.scale()) is not Parameter:
        scale = {r'$\sigma_{}$'.format(a): round(gpd_fit.scale.param_dict[a], 2) for a in gpd_fit.scale.param_dict}
        depending_on = list([g.name for g in gpd_fit.scale.f.args])
        for k in scale:
            text_title += f'{k} = {scale[k]}, '
        text_title += f'depending on variable(s) '
        for l in depending_on:
            l = l.replace('_', ' ')
            text_title += r'\it{' + l + r'}' + ', '
    else:
        scale = round(gpd_fit.scale(), 2)
        text_title += r'$\sigma$=' + str(scale) + ', '
    text_title += '\n'
    if type(gpd_fit.shape()) is not Parameter:
        shape = {r'$\xi_{}$'.format(a): round(gpd_fit.shape.param_dict[a], 2) for a in gpd_fit.shape.param_dict}
        depending_on = list([g.name for g in gpd_fit.shape.f.args])
        for k in shape:
            text_title += f'{k} = {shape[k]}, '
        text_title += f'depending on variable(s) '
        for l in depending_on:
            l = l.replace('_', ' ')
            text_title += r'\it{' + l + r'}' + ', '
    else:
        shape = round(gpd_fit.shape(), 2)
        text_title += r'$\xi$=' + str(shape)
    if text_title.endswith(', '):
        text_title = text_title[:-2]
    threshold_text = str(tuple(threshold)) if hasattr(threshold, '__len__') else str(threshold)
    plt.scatter(empirical_cdf, quantiles_exp, s=5, color="navy")
    plt.plot(empirical_cdf, empirical_cdf, label=f"$x=y$", color="navy")
    plt.legend()
    plt.title(
        "QQ Plot of Exceedances over threshold " + threshold_text + " vs GPD distribution with parameters:\n" + text_title)
    plt.xlabel(f"Empirical ({n} observations)")
    plt.ylabel("GPD distribution")
    plt.tight_layout()
    plt.savefig(f"{path_to_figure}/GPD_fit_exceedances.png")
    plt.clf()

def GEVQQPlot(data: pd.DataFrame, gev_fit: Distribution, path_to_figure: str):
    empirical_cdf = pd.Series(data).quantile(np.linspace(0.01, 0.99, len(data)))
    n = len(data)
    quantiles_exp = gev_fit.inverse_cdf(empirical_cdf.index)
    text_title = ''
    if type(gev_fit.loc()) is not Parameter:
        loc = {r'$\mu_{}$'.format(a): round(gev_fit.loc.param_dict[a], 2) for a in gev_fit.loc.param_dict}
        depending_on = list([g.name for g in gev_fit.loc.f.args])
        for k in loc:
            text_title += f'{k} = {loc[k]}, '
        text_title += f'depending on variables(s) '
        for l in depending_on:
            l = l.replace('_', ' ')
            text_title += r'\it{' + l + r'}' + ', '
    else:
        loc = round(gev_fit.loc(), 2)
        text_title += r'$\mu$=' + str(loc) + ', '
    text_title += '\n'
    if type(gev_fit.scale()) is not Parameter:
        scale = {r'$\sigma_{}$'.format(a): round(gev_fit.scale.param_dict[a], 2) for a in gev_fit.scale.param_dict}
        depending_on = list([g.name for g in gev_fit.scale.f.args])
        for k in scale:
            text_title += f'{k} = {scale[k]}, '
        text_title += f'depending on variable(s) '
        for l in depending_on:
            l = l.replace('_', ' ')
            text_title += r'\it{' + l + r'}' + ', '
    else:
        scale = round(gev_fit.scale(), 2)
        text_title += r'$\sigma$=' + str(scale) + ', '
    text_title += '\n'
    if type(gev_fit.shape()) is not Parameter:
        shape = {r'$\xi_{}$'.format(a): round(gev_fit.shape.param_dict[a], 2) for a in gev_fit.shape.param_dict}
        depending_on = list([g.name for g in gev_fit.shape.f.args])
        for k in shape:
            text_title += f'{k} = {shape[k]}, '
        text_title += f'depending on variable(s) '
        for l in depending_on:
            l = l.replace('_', ' ')
            text_title += r'\it{' + l + r'}' + ', '
    else:
        shape = round(gev_fit.shape(), 2)
        text_title += r'$\xi$=' + str(shape)
    if text_title.endswith(', '):
        text_title = text_title[:-2]
    plt.scatter(empirical_cdf, quantiles_exp, s=5, color="navy")
    plt.plot(empirical_cdf, empirical_cdf, label=f"$x=y$", color="navy")
    plt.legend()
    plt.title("QQ Plot of Maxima vs GEV distribution with parameters:\n" + text_title)
    plt.xlabel(f"Empirical ({n} observations)")
    plt.ylabel("GEV distribution")
    plt.tight_layout()
    plt.savefig(f"{path_to_figure}/GEV_fit_maxima.png")
    plt.clf()

def ExceedancesOverThresholdPerPeriodPlot(data: pd.DataFrame, path_to_figure: str):
    for p, c in zip(data["period"].unique(), ["salmon", "royalblue", "navy"]):
        above_t = data[(data["period"] == p) & (data["data"] >= data["threshold"])]
        plt.scatter(x=above_t["year"].astype(int), y=above_t["data"], c=c, s=15, label=p)
    plt.legend()
    plt.savefig(f"{path_to_figure}/exceedances_per_period.png")
    plt.clf()

def ConsecutiveDaysAboveValuePerYearPlot(data: pd.DataFrame, value: float, path_to_figure: str):
    above_value = data[data["data"] >= value].assign(timedelta=lambda x: x["days_since_start"].diff())
    consecutive = above_value[above_value["timedelta"] == 1].assign(
        timedelta=lambda x: x["days_since_start"].diff()).fillna(0.) \
        .assign(subgroup=lambda x: (x["timedelta"] != x["timedelta"].shift(1)).cumsum()).groupby(
        ["year", "subgroup"]).agg({"data": "count"})
    consecutive = consecutive + 1
    consecutive_mean_per_year = consecutive.groupby(level="year").agg("mean")
    quantile_inf = consecutive.groupby(level="year").agg(lambda x: np.quantile(x, 0.05))
    quantile_sup = consecutive.groupby(level="year").agg(lambda x: np.quantile(x, 0.95))
    plt.scatter(consecutive_mean_per_year.index, consecutive_mean_per_year, label="Mean", color="salmon", s=7)
    plt.vlines(consecutive_mean_per_year.index, quantile_inf, quantile_sup, label=r'5\% Quantiles', alpha=0.6,
               color="salmon")
    plt.legend()
    plt.title(f"Mean Number of Consecutive Days Above {value} per Year.")
    plt.savefig(f"{path_to_figure}/mean_nb_days_cons_above_{value}_year.png")
    plt.clf()

def ConsecutiveDaysUnderValuePerYearPlot(data: pd.DataFrame, value: float, path_to_figure: str):
    under_value = data[data["data"] <= value].assign(timedelta=lambda x: x["days_since_start"].diff())
    consecutive = under_value[under_value["timedelta"] == 1].assign(
        timedelta=lambda x: x["days_since_start"].diff()).fillna(0.) \
        .assign(subgroup=lambda x: (x["timedelta"] != x["timedelta"].shift(1)).cumsum()).groupby(
        ["year", "subgroup"]).agg({"data": "count"})
    consecutive = consecutive + 1
    consecutive_mean_per_year = consecutive.groupby(level="year").agg("mean")
    quantile_inf = consecutive.groupby(level="year").agg(lambda x: np.quantile(x, 0.05))
    quantile_sup = consecutive.groupby(level="year").agg(lambda x: np.quantile(x, 0.95))
    plt.scatter(consecutive_mean_per_year.index, consecutive_mean_per_year, label="Mean", color="salmon", s=7)
    plt.vlines(consecutive_mean_per_year.index, quantile_inf, quantile_sup, label=r'5\% Quantiles', alpha=0.6,
               color="salmon")
    plt.legend()
    plt.title(f"Mean Number of Consecutive Days Under {value} per Year.")
    plt.savefig(f"{path_to_figure}/mean_nb_days_cons_under_{value}_year.png")
    plt.clf()

### INTER EXCEEDANCES DIAGNOSTIC PLOTS, CLUSTERING ###
def MeanInterExceedancePerYear(data: pd.DataFrame, path_to_figure: str):
    """
    Plots the observed inter-exceedance times per year, useful to visualize an increase or decrease in the distance between two extreme events.
    :param data: Dataframe pandas with columns "data", "threshold" (possibly seasonal or periodical), "days_since_start" (of the full period) and "year".
    :param path_to_figure: Path to save the figure.
    :return: Plots diagnostic graphs.
    """
    data = data.dropna().reset_index()
    empty_years = []
    mean_year = []
    quantile_sup_year = []
    quantile_inf_year = []
    for year in data["year"].unique():
        try:
            iat_days = data[(data["year"] == year) & (data["data"] >= data["threshold"])][
                "days_since_start"].diff().dropna()
            quantile_sup_year.append(np.quantile(iat_days, q=0.99))
            quantile_inf_year.append(np.quantile(iat_days, q=0.01))
            mean_year.append(iat_days.mean())
        except:
            empty_years.append(year)
    not_y = [y for y in data["year"].unique() if y not in empty_years]
    plt.scatter(not_y, mean_year, s=7, color="salmon", label="Mean")
    plt.vlines(not_y, quantile_inf_year, quantile_sup_year, alpha=0.6, color="salmon", label=r'10\% Quantiles')
    plt.title("Inter-exceedance time per year")
    plt.legend()
    plt.savefig(f"{path_to_figure}/inter_exceedances_per_year.png")
    plt.clf()

def QQPlotExponentialforInterExceedanceTimes(data: Union[pd.Series, np.array], path_to_figure: str):
    empirical_cdf = pd.Series(data).quantile(np.arange(0., 1., 0.02))
    exp = Exponential.fit(data)
    quantiles_exp = exp.inverse_cdf(empirical_cdf.index)
    plt.scatter(empirical_cdf, quantiles_exp, s=5, color="navy")
    #plt.plot(empirical_cdf, empirical_cdf, label=f"$x=y$", color="navy")
    #plt.legend()
    plt.title("QQ Plot of positive spacing vs Exponential distribution")
    plt.xlabel("Empirical")
    plt.ylabel(r"Exponential with parameter $\lambda$" + f"= {round(exp.rate(), 2)}")
    plt.savefig(f"{path_to_figure}/Exponential fit for IAT.png")
    plt.clf()


def extremogram_loop(h_range, indices_exceedances):
    counts = []
    for h in h_range:
        counts.append(len(np.intersect1d(indices_exceedances + h, indices_exceedances)) / len(indices_exceedances))
    return counts


def ExtremogramPlotHomoPoissonVsHawkes(data: pd.DataFrame, h_range: Union[List, np.array], path_to_figure: str):
    """
    Plots the observed extremogram (ie proba of the observation in t+h to be an exceedance knowing that the observation in t was one.
    Compares the estimate using an homogeneous poisson process vs a Hawkes process.
    :param data: Dataframe pandas containing the columns "data" for the variable of interest, "threshold" for the (possibly seasonal) threshold(s) and "days_since_start" which is a non-normalized version
    of the time in days unit.
    :param h_range: Range for the h parameter to vary through.
    :param path_to_figure: Path to plot the extremogram.
    :return: Plots diagnostic graphs.
    """
    data_extremogram = data[["data", "days_since_start", "threshold"]]
    data_extremogram = data_extremogram[data_extremogram["data"] >= data["threshold"]].assign(
        iat=lambda x: x["days_since_start"].diff()).fillna(0.)
    if UEHP is not None:
        uv = UEHP()
        uv.fit(np.array(data_extremogram["days_since_start"]))
        mu, alpha, theta = uv.get_params()
    else:
        mu = len(data_extremogram)/len(data)
        alpha=0
        theta=0
    e = Exponential.fit(data_extremogram["iat"], x0=(0., mu, alpha, theta),
                        rate=kernels.hawkes_with_exp_kernel(np.array(data_extremogram["days_since_start"])))
    if PP is not None:
        pp = PP()
        pp.fit(np.array(data_extremogram["days_since_start"]))
        rate = pp.get_params()
    else:
       rate = len(data_extremogram)/len(data)
    e2 = Exponential(rate=rate)
    exceedances_hp = []
    for i in range(1000):
        exceedances_hp.append(e.rvs(len(data_extremogram)).cumsum())
    exceedances_pp = []
    for i in range(1000):
        exceedances_pp.append(e2.rvs(len(data_extremogram)).cumsum())
    indices_exceedances = list(data_extremogram["days_since_start"])
    indices_exceedances = [int(round(e, 0)) for e in indices_exceedances]
    extremogram_realized = extremogram_loop(h_range, indices_exceedances)
    extremogram_realized = pd.Series(extremogram_realized, index=h_range)
    count_hp = []
    count_pp = []
    for i in range(len(exceedances_hp)):
        # HP
        ex_hp = exceedances_hp[i]
        indices_exceedances = [int(round(e, 0)) for e in ex_hp]
        local_count_hp = extremogram_loop(h_range, indices_exceedances)
        local_count_hp = pd.Series(local_count_hp, index=h_range)
        count_hp.append(local_count_hp)
        # PP
        ex_pp = exceedances_pp[i].copy()
        indices_exceedances = [int(round(e, 0)) for e in ex_pp]
        local_count_pp = extremogram_loop(h_range, indices_exceedances)
        local_count_pp = pd.Series(local_count_pp, index=h_range)
        count_pp.append(local_count_pp)
    count_hp = pd.concat(count_hp, axis=1)
    count_pp = pd.concat(count_pp, axis=1)

    mean_hp = count_hp.mean(axis=1)
    quantile_inf_hp = np.quantile(count_hp, q=0.01, axis=1)
    quantile_sup_hp = np.quantile(count_hp, q=0.99, axis=1)
    mean_pp = count_pp.mean(axis=1)
    quantile_inf_pp = np.quantile(count_pp, q=0.01, axis=1)
    quantile_sup_pp = np.quantile(count_pp, q=0.99, axis=1)
    plt.plot(h_range, mean_pp, label="Poisson Process Simulated", color="salmon")
    plt.fill_between(x=h_range, y1=quantile_inf_pp, y2=quantile_sup_pp, alpha=0.2, color="salmon")
    plt.plot(h_range, mean_hp, label="Hawkes Process Simulated", color="royalblue")
    plt.fill_between(x=h_range, y1=quantile_inf_hp, y2=quantile_sup_hp, alpha=0.3, color="royalblue")
    plt.bar(x=h_range, height=extremogram_realized, label="Empirical", width=0.8, color="slategrey", alpha=0.6)
    plt.title(f"Extremogram")
    plt.xlabel(r"$h$")
    plt.ylabel(r"$\pi_h(u)$")
    plt.legend()
    plt.savefig(f"{path_to_figure}/extremogram.png")
    plt.clf()
    return e, e2

def loop_mean_cluster_size(indices_exceedances, length_observation_period, block_sizes):
    local_count = []
    for bs in block_sizes:
        hp = np.histogram(indices_exceedances,
                          bins=[bs * i for i in range(math.ceil(length_observation_period / bs))])[0]
        hp = hp[hp > 0]
        local_count.append(np.mean(hp))
    local_count = pd.Series(local_count, index=block_sizes)
    return local_count


def MeanClusterSizeHomoPoissonVsHawkes(data: pd.DataFrame, block_sizes: Union[List, np.array], path_to_figure: str):
    """
    Estimates the extremal intex by computing the mean cluster size considering blocks of sizes pre-defined in the parameter block_sizes, given that
    the block contains an exceedance (ie the mean of pi, the distribution of the extremal index).
    Compares the estimate using an homogeneous poisson process vs a Hawkes process.
    :param data: Dataframe pandas containing the columns "data" for the variable of interest, "threshold" for the defined (possibly seasonal) threshold(s) and "days_since_start" which is a non-normalized version
    of the time in days unit.
    :param block_sizes: Range for cluster sizes.
    :param path_to_figure: Path to save the figure.
    :return: Plots diagnostic graphs.
    """
    data_extremal_index = data[["data", "days_since_start", "threshold"]]
    data_extremal_index = data_extremal_index[data_extremal_index["data"] >= data_extremal_index["threshold"]].assign(
        iat=lambda x: x["days_since_start"].diff()).fillna(0.)
    # Empirical
    empirical_mean_cluster_size = []
    for bs in block_sizes:
        exceedances_clusters = np.histogram(data_extremal_index["days_since_start"],
                                            bins=[bs * i for i in range(math.ceil(len(data) / bs))])[0]
        exceedances_clusters = exceedances_clusters[
            exceedances_clusters > 0]  # given that the block contains an exceedance
        empirical_mean_cluster_size.append(np.mean(exceedances_clusters))
    empirical_mean_cluster_size = pd.Series(empirical_mean_cluster_size, index=block_sizes)

    # Simulated
    if UEHP is not None:
        uv = UEHP()
        uv.fit(np.array(data_extremal_index["days_since_start"]))
        mu, alpha, theta = uv.get_params()
    else:
        mu = len(data_extremal_index)/len(data)
        alpha=0
        theta=0
    e = Exponential.fit(data_extremal_index["iat"], x0=(0., mu, alpha, theta),
                        rate=kernels.hawkes_with_exp_kernel(np.array(data_extremal_index["days_since_start"])))
    if PP is not None:
        pp = PP()
        pp.fit(np.array(data_extremal_index["days_since_start"]))
        rate = pp.get_params()
    else:
       rate = len(data_extremal_index)/len(data)
    e2 = Exponential(rate=rate)
    exceedances_hp = []
    for i in range(1000):
        exceedances_hp.append(e.rvs(len(data_extremal_index)).cumsum())
    exceedances_pp = []
    for i in range(1000):
        exceedances_pp.append(e2.rvs(len(data_extremal_index)).cumsum())
    hp_cluster_sizes = []
    pp_cluster_sizes = []
    for i in range(len(exceedances_hp)):
        # HP
        ex_hp = exceedances_hp[i].copy()
        local_count_hp = loop_mean_cluster_size(ex_hp, data["days_since_start"].max(), block_sizes)
        hp_cluster_sizes.append(local_count_hp)

        # PP
        ex_pp = exceedances_pp[i].copy()
        local_count_pp = loop_mean_cluster_size(ex_pp, data["days_since_start"].max(), block_sizes)
        pp_cluster_sizes.append(local_count_pp)

    hp_cluster_sizes = pd.concat(hp_cluster_sizes, axis=1)
    pp_cluster_sizes = pd.concat(pp_cluster_sizes, axis=1)

    x = block_sizes
    mean_hp = hp_cluster_sizes.mean(axis=1)
    quantile_inf_hp = np.quantile(hp_cluster_sizes, q=0.01, axis=1)
    quantile_sup_hp = np.quantile(hp_cluster_sizes, q=0.99, axis=1)
    mean_pp = pp_cluster_sizes.mean(axis=1)
    quantile_inf_pp = np.quantile(pp_cluster_sizes, q=0.01, axis=1)
    quantile_sup_pp = np.quantile(pp_cluster_sizes, q=0.99, axis=1)
    plt.plot(x, mean_pp, label="Poisson Process Simulated", color="salmon")
    plt.fill_between(x=x, y1=quantile_inf_pp, y2=quantile_sup_pp, alpha=0.2, color="salmon")
    plt.plot(x, mean_hp, label="Hawkes Process Simulated", color="royalblue")
    plt.fill_between(x=x, y1=quantile_inf_hp, y2=quantile_sup_hp, alpha=0.3, color="royalblue")
    plt.bar(x=x, height=empirical_mean_cluster_size, label="Empirical", width=.3, color="slategrey", alpha=0.6)
    plt.title(f"Mean Cluster Size")
    plt.xlabel(r"Block size $r$ (days)")
    plt.ylabel(r"Number of exceedances per block $\theta^{-1}_r(u)$")
    plt.legend(loc="upper left")
    plt.savefig(f"{path_to_figure}/mean_cluster_size.png")
    plt.clf()
    return e, e2

def CumNumberOfExceedancesHomoPoissonVsHawkes(data: pd.DataFrame,
                                              length_total_period: Union[int, float],
                                              origin: pd.datetime,
                                              path_to_figure: str):
    """
    Computes the observed and simulated cumulative number of exceedances and compares a homogeneous poisson process estimate with the one obtained
    using a Hawkes process for modeling inter-exceedance times.
    :param data: Dataframe pandas containing the columns "data", "threshold" and "time" (normalized between 0 and 1) for less numerical errors.
    :param length_total_period: Length of the total period for scaling (graphical purpose only, unlike in previous graphs
    comparing Hawkes and HomoPoisson where a discrete scale is preferred vs a normalized "continuous" one between 0 and 1 due
    to precision concerns using input block sizes and deltas), same unit as inter-exceedance times.
    :param origin: The start date for scaling purposes (graphical).
    :param path_to_figure: Path to save the figure.
    :return: Plots diagnostic graphs.
    """
    data_gpd = data.set_index("time")[["data", "threshold"]]
    data_gpd = data_gpd[data_gpd["data"] >= data_gpd["threshold"]].reset_index().assign(
        iat=lambda x: x["time"].diff()).fillna(0.)
    if UEHP is not None:
        uv = UEHP()
        uv.fit(np.array(data_gpd["time"]))
        mu, alpha, theta = uv.get_params()
    else:
        mu = 1/(len(data_gpd)/len(data))
        alpha=0
        theta=0
    e = Exponential.fit(data_gpd["iat"], x0=(0., mu, alpha, theta),
                        rate=kernels.hawkes_with_exp_kernel(np.array(data_gpd["time"])))
    if PP is not None:
        pp = PP()
        pp.fit(np.array(data_gpd["time"]))
        rate = pp.get_params()
    else:
       rate = 1/(len(data_gpd)/len(data))
    e2 = Exponential(rate=rate)
    simulations = []
    for _ in range(1000):
        simulations.append(pd.DataFrame(np.cumsum(e.rvs(len(data_gpd))), columns=[_]))
    g = pd.concat(simulations, axis=1)
    mean = g.mean(axis=1)
    std = g.std(axis=1)

    r = data.set_index("time").assign(bool=lambda x: (x["data"] >= x["threshold"]).astype(int))["bool"]
    r = r[r == True].cumsum().reset_index().set_index("bool")["time"]

    simulations_pp = []
    for _ in range(1000):
        sample_pp = np.cumsum(e2.rvs(len(data_gpd)))
        simulations_pp.append(pd.DataFrame(sample_pp, columns=[_]))
    simu_pp = pd.concat(simulations_pp, axis=1)
    mean_pp = simu_pp.mean(axis=1)
    std_pp = simu_pp.std(axis=1)

    def swap_axes(s):
        return pd.Series(s.index.values, index=s)

    r_ = swap_axes(r)
    mean_ = swap_axes(mean)
    mean_pp_ = swap_axes(mean_pp)
    x = pd.to_datetime(length_total_period * data_gpd["time"],
                       unit="d", origin=origin)
    x1 = pd.to_datetime(length_total_period * (mean + std),
                        unit="d", origin=origin)
    x2 = pd.to_datetime(length_total_period * (mean - std),
                        unit="d", origin=origin)
    x1_pp = pd.to_datetime(length_total_period * (mean_pp + std_pp),
                           unit="d", origin=origin)
    x2_pp = pd.to_datetime(length_total_period * (mean_pp - std_pp),
                           unit="d", origin=origin)
    mean_.index = pd.to_datetime(length_total_period * mean,
                                 unit="d", origin=origin)
    mean_pp_.index = pd.to_datetime(length_total_period * mean_pp,
                                    unit="d", origin=origin)
    r_.index = x
    plt.plot(mean_pp_, label="Poisson Process Simulated", color="salmon")
    plt.fill_betweenx(y=mean_pp_, x1=x1_pp, x2=x2_pp, color="salmon", alpha=0.2)
    plt.plot(mean_, label="Hawkes Process Simulated", color="royalblue")
    plt.fill_betweenx(y=mean_, x1=x1, x2=x2, color="royalblue", alpha=0.2)
    plt.plot(r_, label="Empirical", color="slategrey")
    plt.title("Cumulative Exceedances over Threshold")
    plt.ylabel("Number of Exceedances")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"{path_to_figure}/cumulative_nb_exceedances.png")
    plt.clf()
    return e, e2


def KgapsMethodDiagnosticPlots(range_K: Union[List, np.array],
                               inter_exceedance_times: pd.Series,
                               proba_of_exceedance: float,
                               path_to_figure: str):
    """

    :param range_K: Range in which the parameter K of the model varies; should be the same unit as the inter-exceedance times
    :param inter_exceedance_times: Observed inter-exceedance times
    :param proba_of_exceedance: Normalizing factor corresponding to 1-F(un) in the model, proba of exceeding the defined threshold
    :param path_to_figure: Path to save the plots
    :return: Plots the diagnostic graphs
    """
    Ks = range_K
    thetas = []
    CIs = []
    for K in Ks:
        iet_normalised_to_cluster_distance = np.clip(inter_exceedance_times - K, 0., a_max=None) * proba_of_exceedance
        mem = MixtureExponentialModel.fit(iet_normalised_to_cluster_distance)
        theta = mem.theta()
        ll = Likelihood(mem, iet_normalised_to_cluster_distance)
        CI = [ll.profiles["theta"]["theta"].min(), ll.profiles["theta"]["theta"].max()]
        thetas.append(theta)
        CIs.append(CI)
        freq, bins = np.histogram(iet_normalised_to_cluster_distance, bins="auto")
        freq = freq / len(iet_normalised_to_cluster_distance)
        x = (bins[:-1] + bins[1:]) / 2
        empirical_cdf = pd.Series(
            iet_normalised_to_cluster_distance[iet_normalised_to_cluster_distance > 0]).quantile(np.arange(0.01, 0.99, 0.01))
        quantiles_exp = Exponential(rate=theta).inverse_cdf(empirical_cdf.index)
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(1, 2)
        ax = []
        for i in range(1):
            for j in range(2):
                axe = fig.add_subplot(gs[i, j])
                ax.append(axe)
        plt.subplots_adjust(wspace=0.2)
        ax0, ax1 = ax
        ax0.bar(x, freq, color="navy")
        ax0.set_title(f"Spacings between exceedances distribution for K={round(K, 1)} days")
        ax0.set_xlabel("Interval")
        ax0.set_ylabel("Frequency")
        ax1.scatter(empirical_cdf, quantiles_exp, s=5, color="navy")
        #ax1.plot(empirical_cdf, empirical_cdf, label=f"$x=y$", color="navy")
        #ax1.legend()
        ax1.set_title("QQ Plot of positive spacing vs Exponential distribution")
        ax1.set_xlabel("Empirical")
        ax1.set_ylabel(r"Exponential with $\theta$" + f"= {round(theta, 2)}")
        fig.savefig(f"{path_to_figure}/spacings_summary_{round(K, 1)}.png")
    serie = pd.Series([float(t) for t in thetas], index=Ks)
    CI = pd.DataFrame(CIs)
    plt.clf()
    plt.scatter(serie.index, serie, marker='x', color="navy")
    plt.vlines(serie.index, CI[0], CI[1], color="navy")
    plt.xlabel("Inter-cluster interval $K$ (days)")
    plt.ylabel(r"$\hat{\theta}$")
    plt.title("Extremal index estimate")
    plt.savefig(f"{path_to_figure}/K-gaps_extremal_index_estimate.png")
    plt.clf()


def K_gaps_estimate_of_inter_exceedances(X: Union[pd.Series, pd.DataFrame],
                                         threshold: Union[float, int, str],
                                         K: Union[float, int] = None,
                                         data_column_name: str = None):
    """
    Warning: estimate based on exponential inter-exceedance times, meaning that exceedances are arriving at times
    independant from history (memoryless property of the exponential).
    :param X: dataset with index the a time info of any sort (can also be a standard index if all observations
    are observed in a stationary way, eg with constant spacing)
    :param K: min spacing of exceedances from separate clusters (space between clusters)
    :param threshold: threshold delimiting exceedances. If a string is passed, name of the column of X where the thresholds are.
    :param data_column_name: if X is a dataframe, column with data
    :return: K-gaps estimate for theta, the extremal index
    """
    data = X.copy()
    if isinstance(X, pd.DataFrame):
        if len(X.columns) > 1 and (data_column_name is None or data_column_name not in X.columns):
            raise TypeError(
                "X is a dataframe of more than one columns, yet no column name has been specified for the variable of interest.")
        elif len(X.columns) > 1:
            data = X[data_column_name]
        elif len(X.columns) == 1:
            data = X[X.columns[0]]
        if isinstance(threshold, str):
            if not threshold in X.columns:
                raise NameError(
                    f"{threshold} has been passed as value for the threshold column of X, yet X does not contain a column with that name.")
            threshold = X[threshold]
    exceedances = data[data >= threshold]
    indices_exceedances = np.array(exceedances.index)
    inter_exceedance_times = np.diff(indices_exceedances)
    if K is None:
        K = np.mean(inter_exceedance_times)
    iet_normalised_to_cluster_distance = np.clip(inter_exceedance_times - K, 0., a_max=None)
    n_positive = len(iet_normalised_to_cluster_distance[iet_normalised_to_cluster_distance > 0.])
    n0 = len(iet_normalised_to_cluster_distance) - n_positive

    def opposite_log_likelihood(theta: float):
        normalising_factor = len(exceedances) / len(X)
        positive_spacings = normalising_factor * iet_normalised_to_cluster_distance[
            iet_normalised_to_cluster_distance > 0.]
        zero_mass = n0 * np.log(1 - theta)
        exponential_mass = 2 * n_positive * np.log(theta) - theta * np.sum(positive_spacings)
        return -(zero_mass + exponential_mass)

    estimate_theta = minimize(opposite_log_likelihood, x0=np.array(0.99), method="Nelder-Mead").x
    return estimate_theta
