import warnings
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["text.usetex"] = True

from pykelihood.distributions import GEV, Distribution, Exponential, Uniform
from pykelihood.profiler import Profiler

warnings.filterwarnings("ignore")


def get_quantiles_and_confidence_intervals_uniform_scale(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ci_confidence=0.99,
):
    ll = Profiler(fit, data, inference_confidence=ci_confidence)
    min_max = []
    levels = np.linspace(0.01, 0.99, 100)
    for level in levels:
        metric = lambda x: pd.Series(x.cdf(data)).quantile(level)
        CI = ll.confidence_interval(metric)
        min_max.append(CI)
    min_max = pd.DataFrame(
        min_max, columns=["lower_bound", "upper_bound"], index=levels
    )
    empirical = pd.Series(ll.optimum[0].cdf(data)).quantile(levels)
    theoretical = Uniform(0, 1).inverse_cdf(levels)
    return theoretical, empirical, min_max["lower_bound"], min_max["upper_bound"]


def get_quantiles_and_confidence_intervals(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ci_confidence=0.99,
):
    ll = Profiler(fit, data, inference_confidence=ci_confidence)
    min_max = []
    levels = np.linspace(0.01, 0.99, 100)
    empirical = pd.Series(data).quantile(levels)
    theoretical = ll.optimum[0].inverse_cdf(levels)
    for level in levels:
        metric = lambda x: x.inverse_cdf(level)
        CI = ll.confidence_interval(metric)
        min_max.append(CI)
    min_max = pd.DataFrame(
        min_max, columns=["lower_bound", "upper_bound"], index=levels
    )
    return theoretical, empirical, min_max["lower_bound"], min_max["upper_bound"]


def pp_plot(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ax=plt.gca(),
    path_to_figure: str = None,
    figure_name="pp_plot",
    ci_confidence=0.99,
):
    (
        theoretical,
        empirical,
        lower_bound,
        upper_bound,
    ) = get_quantiles_and_confidence_intervals_uniform_scale(fit, data, ci_confidence)
    n = len(data)
    ax.scatter(theoretical, empirical, s=5, color="navy")
    ax.plot(theoretical, theoretical, label=f"$x=y$", color="navy")
    ax.fill_between(theoretical, lower_bound, upper_bound, alpha=0.2, color="navy")
    ax.legend()
    ax.set_xlabel(f"Theoretical quantiles ({n} observations)")
    ax.set_label("Empirical quantiles")
    ax.set_title("PP Plot")
    plt.tight_layout()
    if path_to_figure is not None:
        plt.savefig(f"{path_to_figure}/{figure_name}.png")
    return ax


def qq_plot(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ax=plt.gca(),
    path_to_figure: str = None,
    figure_name="qq_plot",
    ci_confidence=0.99,
):
    (
        theoretical,
        empirical,
        lower_bound,
        upper_bound,
    ) = get_quantiles_and_confidence_intervals(fit, data, ci_confidence)
    n = len(data)
    ax.scatter(theoretical, empirical, s=5, color="navy")
    ax.plot(theoretical, theoretical, label=f"$x=y$", color="navy")
    ax.fill_betweenx(
        y=empirical, x1=lower_bound, x2=upper_bound, alpha=0.2, color="navy"
    )
    ax.legend()
    ax.set_xlabel(f"Theoretical quantiles ({n} observations)")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("QQ Plot")
    plt.tight_layout()
    if path_to_figure is not None:
        plt.savefig(f"{path_to_figure}/{figure_name}.png")
    return ax


def qq_plot_exponential_scale(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ax=plt.gca(),
    path_to_figure: str = None,
    figure_name="qq_plot",
    ci_confidence=0.99,
):
    (
        theoretical,
        empirical,
        lower_bound,
        upper_bound,
    ) = get_quantiles_and_confidence_intervals_uniform_scale(fit, data, ci_confidence)
    unit_exp = Exponential()
    theo_exp = unit_exp.inverse_cdf(theoretical)
    empi_exp = unit_exp.inverse_cdf(empirical)
    lb_exp = unit_exp.inverse_cdf(lower_bound)
    ub_exp = unit_exp.inverse_cdf(upper_bound)
    n = len(data)
    ax.scatter(theo_exp, empi_exp, s=5, color="navy")
    ax.plot(theo_exp, theo_exp, label=f"$x=y$", color="navy")
    ax.fill_betweenx(y=empi_exp, x1=lb_exp, x2=ub_exp, alpha=0.2, color="navy")
    ax.legend()
    ax.set_xlabel(f"Theoretical unit Exponential quantiles ({n} observations)")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("QQ Plot: Unit Exponential Scale")
    plt.tight_layout()
    if path_to_figure is not None:
        plt.savefig(f"{path_to_figure}/{figure_name}.png")
    return ax


def qq_plot_frechet_scale(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ax=plt.gca(),
    path_to_figure: str = None,
    figure_name="qq_plot",
    ci_confidence=0.99,
):
    (
        theoretical,
        empirical,
        lower_bound,
        upper_bound,
    ) = get_quantiles_and_confidence_intervals_uniform_scale(fit, data, ci_confidence)
    unit_frechet = GEV(1, 1, -1)
    theo_fr = unit_frechet.inverse_cdf(theoretical)
    empi_fr = unit_frechet.inverse_cdf(empirical)
    lb_fr = unit_frechet.inverse_cdf(lower_bound)
    ub_fr = unit_frechet.inverse_cdf(upper_bound)
    n = len(data)
    ax.scatter(theo_fr, empi_fr, s=5, color="navy")
    ax.plot(theo_fr, theo_fr, label=f"$x=y$", color="navy")
    ax.fill_betweenx(y=empi_fr, x1=lb_fr, x2=ub_fr, alpha=0.2, color="navy")
    ax.legend()
    ax.set_xlabel(r"Theoretical unit Fr\'echet quantiles ({} observations)".format(n))
    ax.set_ylabel("Empirical quantiles")
    ax.set_title(r"QQ Plot: Unit Fr\'echet Scale")
    plt.tight_layout()
    if path_to_figure is not None:
        plt.savefig(f"{path_to_figure}/{figure_name}.png")
    return ax
