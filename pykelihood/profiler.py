from __future__ import annotations

import math
import warnings
from itertools import count
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2

from pykelihood.distributions import Distribution
from pykelihood.metrics import opposite_log_likelihood

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

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
            optimization_method="Nelder-Mead",
            x0=None,
    ):
        """
        Initialize the Profiler.

        Parameters
        ----------
        distribution : Distribution
            Distribution on which the inference is based.
        data : pd.Series
            Variable of interest.
        score_function : Callable, optional
            Function used for optimization, by default opposite_log_likelihood.
        name : str, optional
            Name of the profile if it needs to be compared to other score functions, by default "Standard".
        inference_confidence : float, optional
            Desired confidence for intervals, by default 0.99.
        single_profiling_param : optional
            Parameter to fix to create the profiles based on the score function, by default None.
        optimization_method : str, optional
            Method used for optimization, by default "Nelder-Mead".
        x0 : optional
            Initial guess for the optimization, by default None.
        """
        self.name = name
        self.distribution = distribution
        self.data = data
        self.score_function = score_function
        self.inference_confidence = inference_confidence
        self.single_profiling_param = single_profiling_param
        self.optimization_method = optimization_method
        self.x0 = x0

    @cached_property
    def standard_mle(self):
        """
        Compute the standard maximum likelihood estimate (MLE) for the distribution.

        Returns
        -------
        tuple
            A tuple containing the estimate and the log-likelihood value.
        """
        estimate = self.distribution.fit(self.data, method=self.optimization_method)
        ll = -opposite_log_likelihood(estimate, self.data)
        ll = ll if isinstance(ll, float) else ll[0]
        return (estimate, ll)

    @cached_property
    def optimum(self):
        """
        Compute the optimum parameters for the distribution using the score function.

        Returns
        -------
        tuple
            A tuple containing the estimate and the score function value.
        """
        estimate = self.distribution.fit_instance(
            self.data,
            score=self.score_function,
            x0=self.x0,
            scipy_args={"method": self.optimization_method},
        )
        func = -self.score_function(estimate, self.data)
        func = func if isinstance(func, float) else func[0]
        return (estimate, func)

    @cached_property
    def profiles(self):
        """
        Compute the profile likelihoods for the parameters.

        Returns
        -------
        dict
            A dictionary with parameter names as keys and their profile likelihoods as values.
        """
        profiles = {}
        opt, func = self.optimum
        if self.single_profiling_param is not None:
            params = [self.single_profiling_param]
        else:
            params = [n[0] for (v, n) in opt.param_mapping()]
        for name, k in opt.optimisation_param_dict.items():
            if name in params:
                r = float(k)
                b = 2 * 10 ** (math.floor(math.log10(np.abs(r))) - 1)
                range = np.linspace(r - b, r + b, 40)
                profiles[name] = self.test_profile_likelihood(range, name)
        return profiles

    def test_profile_likelihood(self, range_for_param, param):
        """
        Test the profile likelihood for a given parameter over a specified range.

        Parameters
        ----------
        range_for_param : array-like
            Range of values for the parameter.
        param : str
            Name of the parameter.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the filtered parameters and their scores.
        """
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

    def confidence_interval(self, param: str, precision=1e-5) -> Tuple[float, float]:
        """
        Compute the confidence interval for a given parameter.

        Parameters
        ----------
        param : str
            Name of the parameter.
        precision : float, optional
            Precision for the confidence interval calculation, by default 1e-5.

        Returns
        -------
        tuple
            A tuple containing the lower and upper bounds of the confidence interval.
        """
        opt, func = self.optimum
        value_threshold = func - chi2.ppf(self.inference_confidence, df=1) / 2

        def score(x: float):
            new_opt = opt.fit_instance(
                self.data, score=self.score_function, **{param: x}
            )
            return -self.score_function(new_opt, self.data)

        def delta_to_threshold(x: float):
            return score(x) - value_threshold

        def is_outside_conf_interval(x: float):
            return score(x) < value_threshold

        param_value = opt.flattened_param_dict[param].value
        if is_outside_conf_interval(param_value):
            raise RuntimeError("Optimum is outside its own confidence interval?")

        step = 0.1 * abs(param_value)
        lb_underestimate = param_value - step
        ub_overestimate = param_value + step
        for _ in range(1000):
            if is_outside_conf_interval(lb_underestimate):
                break
            lb_underestimate -= step
        else:  # nobreak
            raise RuntimeError("Unable to find confidence interval lower bound")

        for _ in range(1000):
            if is_outside_conf_interval(ub_overestimate):
                break
            ub_overestimate += step
        else:  # nobreak
            raise RuntimeError("Unable to find confidence interval upper bound")

        lb = brentq(delta_to_threshold, lb_underestimate, param_value, xtol=precision)
        ub = brentq(delta_to_threshold, param_value, ub_overestimate, xtol=precision)

        return lb, ub

    # For compatibility
    confidence_interval_bs = confidence_interval
