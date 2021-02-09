import warnings
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["text.usetex"] = True

from pykelihood.distributions import Distribution, Uniform
from pykelihood.stats_utils import Likelihood

warnings.filterwarnings("ignore")


def get_quantiles_and_confidence_intervals_uniform_scale(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ci_confidence=0.99,
):
    ll = Likelihood(fit, data, inference_confidence=ci_confidence)
    min_max = []
    levels = np.linspace(0.01, 0.99, 100)
    for level in levels:
        metric = lambda x: pd.Series(x.cdf(data)).quantile(level)
        CI = ll.confidence_interval_for_specified_metric(metric)
        min_max.append(CI)
    min_max = pd.DataFrame(
        min_max, columns=["lower_bound", "upper_bound"], index=levels
    )
    empirical = pd.Series(ll.mle[0].cdf(data)).quantile(levels)
    theoretical = Uniform(0, 1).inverse_cdf(levels)
    return theoretical, empirical, min_max["lower_bound"], min_max["upper_bound"]


def get_quantiles_and_confidence_intervals(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    ci_confidence=0.99,
):
    ll = Likelihood(fit, data, inference_confidence=ci_confidence)
    min_max = []
    levels = np.linspace(0.01, 0.99, 100)
    empirical = pd.Series(data).quantile(levels)
    theoretical = fit.inverse_cdf(levels)
    for level in levels:
        metric = lambda x: x.inverse_cdf(level)
        CI = ll.confidence_interval_for_specified_metric(metric)
        min_max.append(CI)
    min_max = pd.DataFrame(
        min_max, columns=["lower_bound", "upper_bound"], index=levels
    )
    return theoretical, empirical, min_max["lower_bound"], min_max["upper_bound"]


def qq_plot_uniform_scale(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    path_to_figure: str,
    ci_confidence=0.99,
    figure_name="qq_plot",
):
    (
        theoretical,
        empirical,
        lower_bound,
        upper_bound,
    ) = get_quantiles_and_confidence_intervals_uniform_scale(fit, data, ci_confidence)
    n = len(data)
    plt.scatter(theoretical, empirical, s=5, color="navy")
    plt.plot(theoretical, theoretical, label=f"$x=y$", color="navy")
    plt.fill_between(theoretical, lower_bound, upper_bound, alpha=0.2, color="navy")
    plt.legend()
    plt.xlabel(f"Theoretical quantiles ({n} observations)")
    plt.ylabel("Empirical quantiles")
    plt.tight_layout()
    plt.savefig(f"{path_to_figure}/{figure_name}.png")
    plt.clf()


def qq_plot(
    fit: Distribution,
    data: Union[pd.DataFrame, np.array, pd.Series],
    path_to_figure: str,
    ci_confidence=0.99,
    figure_name="qq_plot",
):
    (
        theoretical,
        empirical,
        lower_bound,
        upper_bound,
    ) = get_quantiles_and_confidence_intervals(fit, data, ci_confidence)
    import matplotlib.pyplot as plt

    n = len(data)
    plt.scatter(theoretical, empirical, s=5, color="navy")
    plt.plot(theoretical, theoretical, label=f"$x=y$", color="navy")
    plt.fill_betweenx(
        y=empirical, x1=lower_bound, x2=upper_bound, alpha=0.2, color="navy"
    )
    plt.legend()
    plt.xlabel(f"Theoretical quantiles ({n} observations)")
    plt.ylabel("Empirical quantiles")
    plt.tight_layout()
    plt.savefig(f"{path_to_figure}/{figure_name}.png")
    plt.clf()
