from __future__ import annotations

import bisect
import math
import warnings
from collections.abc import Sequence
from itertools import count
from typing import Callable, Tuple

import numpy as np
import pandas as pd
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
    ):
        """l

        :param distribution: distribution on which the inference is based
        :param data: variable of interest
        :param score_function: function used for optimisation
        :param name: name (optional) of the profile if it needs to be compared to other score functions
        :param inference_confidence: wanted confidence for intervals
        :param single_profiling_param: parameter that we want to fix to create the profiles based on the score function
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
        estimate = self.distribution.fit_instance(self.data, score=self.score_function)
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
            params = [n[0] for (v, n) in opt.param_mapping()]
        for name, k in opt.optimisation_param_dict.items():
            if name in params:
                r = float(k)
                b = 2 * 10 ** (math.floor(math.log10(np.abs(r))) - 1)
                range = np.linspace(r - b, r + b, 40)
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

    def confidence_interval_bs(self, param: str, precision=1e-5) -> Tuple[float, float]:
        opt, func = self.optimum
        value_threshold = func - chi2.ppf(self.inference_confidence, df=1) / 2

        def is_inside_conf_interval(x: float):
            new_opt = opt.fit_instance(
                self.data, score=self.score_function, **{param: x}
            )
            return -self.score_function(new_opt, self.data) >= value_threshold

        def is_outside_conf_interval(x: float):
            return not is_inside_conf_interval(x)

        param_value = opt.flattened_param_dict[param].value
        if is_outside_conf_interval(param_value):
            return [None, None]

        step = 0.1 * param_value
        lb = param_value - step
        ub = param_value + step
        for _ in range(1000):
            if is_outside_conf_interval(lb):
                break
            lb -= step
        else:  # nobreak
            return [None, None]
        for _ in range(1000):
            if is_outside_conf_interval(ub):
                break
            ub += step
        else:  # nobreak
            return [None, None]

        class LazyMappingList(Sequence):
            def __init__(self, value_mapping, fn: Callable):
                self.mapping = value_mapping
                self.fn = fn

            def __len__(self):
                return len(self.mapping)

            def __getitem__(self, item):
                return self.fn(self.mapping[item])

        mapping = np.linspace(lb, ub, num=int(1 / precision))
        search_size = int(step / ((ub - lb) * precision))
        lb_index = bisect.bisect_left(
            LazyMappingList(mapping, is_inside_conf_interval),
            True,
            lo=0,
            hi=search_size,
        )
        ub_index = (
            bisect.bisect_left(
                LazyMappingList(mapping, is_outside_conf_interval),
                True,
                lo=len(mapping) - search_size,
            )
            - 1
        )

        return mapping[lb_index], mapping[ub_index]

    def confidence_interval(self, metric: Callable[[Distribution], float]):
        """

        :param metric: function depending on the distribution: it can be one of the parameter (ex: lambda x: x.shape() for a parameter called "shape"),
        or a metric relevant to the field of study (ex: the 100-years return level for extreme value analysis by setting lambda x: x.isf(1/100))...
        :return: bounds based on parameter profiles for this metric
        """
        estimates = []
        profiles = self.profiles
        params = list(profiles.keys())
        for param in params:
            columns = [
                n[0]
                for (v, n) in self.optimum[0].param_mapping()
                if n[0] in self.optimum[0].optimisation_param_dict
            ]
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
