import resource
import time
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd

from pykelihood.distributions import CompositionDistribution, Distribution, Uniform
from pykelihood.stats_utils import ConditioningMethod, Likelihood
from pykelihood.stopping_times import StoppingRule

warnings.filterwarnings('ignore')

USE_POOL = False


class SimulationEVD(object):
    def __init__(self, reference: Distribution,
                 historical_sample: np.array,
                 n_iter: int,
                 return_period: int = 200,
                 sample_size: int = 790):
        self.ref_distri = reference
        self.n_iter = n_iter
        self.return_period = return_period
        self.true_return_level = self.ref_distri.isf(1 / return_period)
        self.sample_size = sample_size
        self.historical_sample = historical_sample
        self.rl_estimates = {}
        self.CI = {}

    def __call__(self, *args, **kwargs):
        pass

    def RRMSE(self):
        print(f"Computing the mean RRMSE for the {self.n_iter} samples...")
        RRMSE = {}
        for name in self.rl_estimates.keys():
            RRMSE[name] = {}
            for k in self.rl_estimates[name].keys():
                non_null = [j for j in self.rl_estimates[name][k] if j is not None and j < 1000]
                sqrt = np.sqrt(np.mean((np.array(non_null) - self.true_return_level) ** 2))
                RRMSE[name][k] = (1 / self.true_return_level) * sqrt
        return RRMSE

    def RelBias(self):
        print(f"Computing the mean relative bias for the {self.n_iter} samples...")
        RelBias = {}
        for name in self.rl_estimates.keys():
            RelBias[name] = {}
            for k in self.rl_estimates[name].keys():
                non_null = [j for j in self.rl_estimates[name][k] if j is not None and j < 1000]
                relbias = (1 / self.true_return_level) * np.mean(non_null) - 1
                RelBias[name][k] = relbias
        return RelBias

    def CI_Coverage(self):
        print(f"Computing the mean CI Coverage for the {self.n_iter} samples...")
        CIC = {}
        for name in self.CI.keys():
            CIC[name] = {}
            for k in self.CI[name].keys():
                non_null = [(lb, ub) for (lb, ub) in self.CI[name][k] if
                            lb is not None and ub is not None and ub - lb < 1000]
                bools = [lb <= self.true_return_level <= ub
                         for lb, ub in non_null]
                CIC[name][k] = sum(bools) / len(bools) if bools else 0.
        return CIC

    def CI_Width(self):
        print(f"Computing the mean CI Width for the {self.n_iter} samples...")
        CIW = {}
        for name in self.CI.keys():
            CIW[name] = {}
            for k in self.CI[name].keys():
                non_null = [ub - lb for (lb, ub) in self.CI[name][k] if
                            lb is not None and ub is not None and ub - lb < 1000]
                CIW[name][k] = np.mean(non_null)
        return CIW


class SimulationWithVaryingRefParamsForJointModel(SimulationEVD):
    def __init__(self, reference: CompositionDistribution,
                 historical_sample: np.array,
                 n_iter: int,
                 range_params: Tuple[Sequence, Sequence],
                 return_period: int = 100,
                 sample_size: int = 300):
        super(SimulationWithVaryingRefParamsForJointModel, self).__init__(reference,
                                                                          historical_sample,
                                                                          n_iter,
                                                                          return_period,
                                                                          sample_size)
        self.range_params = range_params
        rl_estimates = {y: {x: [] for x in self.range_params[0]} for y in self.range_params[1]}
        CI = {y: {x: [] for x in self.range_params[0]} for y in self.range_params[1]}
        if USE_POOL:
            pool = Pool(cpu_count())
            results = pool.map(self.__call__, self.range_params[0])
            pool.close()
        else:
            results = [self(x) for x in self.range_params[0]]
        for res in results:
            for name, v in res.items():
                for k, value in v.items():
                    rl_estimates[name][k] = list([p[0] for p in value])
                    CI[name][k] = list([p[1] for p in value])
        self.rl_estimates = rl_estimates
        self.CI = CI

    def __call__(self, x):
        time_start = time.time()
        res = {y: {} for y in self.range_params[1]}
        for y in self.range_params[1]:
            distri = self.ref_distri.with_params((y, x))
            datasets = (pd.Series(distri.rvs(self.sample_size)) for _ in range(self.n_iter))
            res[y][x] = []
            for data in datasets:
                res[y][x].append(to_run_in_parallel(data, distri, self.return_period,
                                                    ("Standard", ConditioningMethod.no_conditioning)))
        time_elapsed = time.time() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))
        return res

    def RRMSE(self):
        print(f"Computing the mean RRMSE for the {self.n_iter} samples...")
        RRMSE = {}
        for x in self.rl_estimates.keys():
            RRMSE[x] = {}
            for y in self.rl_estimates[x].keys():
                true_return_level = self.ref_distri.with_params((x, y)).isf(1 / self.return_period)
                non_null = [j for j in self.rl_estimates[x][y] if j is not None and j < 1000]
                sqrt = np.sqrt(np.mean((np.array(non_null) - true_return_level) ** 2))
                RRMSE[x][y] = (1 / self.true_return_level) * sqrt
        return RRMSE

    def RelBias(self):
        print(f"Computing the mean relative bias for the {self.n_iter} samples...")
        RelBias = {}
        for x in self.rl_estimates.keys():
            RelBias[x] = {}
            for y in self.rl_estimates[x].keys():
                true_return_level = self.ref_distri.with_params((x, y)).isf(1 / self.return_period)
                non_null = [j for j in self.rl_estimates[x][y] if j is not None and j < 1000]
                relbias = (1 / true_return_level) * np.mean(non_null) - 1
                RelBias[x][y] = relbias
        return RelBias

    def CI_Coverage(self):
        print(f"Computing the mean CI Coverage for the {self.n_iter} samples...")
        CIC = {}
        for x in self.CI.keys():
            CIC[x] = {}
            for y in self.CI[x].keys():
                true_return_level = self.ref_distri.with_params((x, y)).isf(1 / self.return_period)
                non_null = [(lb, ub) for (lb, ub) in self.CI[x][y] if
                            lb is not None and ub is not None and ub - lb < 1000]
                bools = [lb <= true_return_level <= ub
                         for lb, ub in non_null]
                CIC[x][y] = sum(bools) / len(bools) if bools else 0.
        return CIC


class SimulationWithStoppingRuleAndConditioning(SimulationEVD):
    def __init__(self, reference: Distribution,
                 historical_sample: np.array,
                 n_iter: int,
                 stopping_rule_func: Callable,
                 conditioning_rules: Sequence[Tuple[str, Callable]],
                 return_periods_for_threshold: Sequence[int],
                 return_period: int = 200,
                 sample_size: int = 790):
        self.stopping_rule_func = stopping_rule_func
        self.conditioning_rules = conditioning_rules
        self.return_periods_for_threshold = return_periods_for_threshold
        super(SimulationWithStoppingRuleAndConditioning, self).__init__(reference,
                                                                        historical_sample,
                                                                        n_iter,
                                                                        return_period,
                                                                        sample_size)
        rl_estimates = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        CI = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        datasets = (pd.Series(np.concatenate([self.historical_sample,
                                              self.ref_distri.rvs(self.sample_size)]))
                    for _ in range(self.n_iter))
        if USE_POOL:
            pool = Pool(cpu_count())
            results = pool.map(self.__call__, datasets)
            pool.close()
        else:
            results = [self(ds) for ds in datasets]
        for res in results:
            for name, v in res.items():
                for k, value in v.items():
                    rl_estimates[name][k].append(value[0])
                    CI[name][k].append(value[1])
        self.rl_estimates = rl_estimates
        self.CI = CI

    def __call__(self, data):
        time_start = time.time()
        res = {name: {} for name, _ in self.conditioning_rules}
        for x in self.return_periods_for_threshold:
            if self.stopping_rule_func == StoppingRule.fixed_to_k:
                k = self.ref_distri.isf(1 / x)
            else:
                k = x
            stopping_rule = StoppingRule(data, self.ref_distri,
                                         k=k, historical_sample_size=10,
                                         func=self.stopping_rule_func)
            data_stopped = stopping_rule.stopped_data()
            std_conditioning = ["Standard", "Excluding Last Obs"]
            crules = [(name, partial(crule, threshold=stopping_rule.threshold())) \
                      for (name, crule) in self.conditioning_rules if name
                      not in std_conditioning] \
                     + [(name, crule) for (name, crule) in self.conditioning_rules if name in std_conditioning]
            f = partial(to_run_in_parallel,
                        *[data_stopped, self.ref_distri, self.return_period])
            for name, c in crules:
                res[name][x] = f((name, c))
        time_elapsed = time.time() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))
        return res

class SimulationWithStoppingRuleAndConditioningForConditionedObservations(SimulationEVD):
    def __init__(self, reference: Distribution,
                 historical_sample: np.array,
                 n_iter: int,
                 conditioning_rules: Sequence[Tuple[str, Callable]],
                 return_periods_for_threshold: Sequence[int],
                 return_period: int = 200,
                 sample_size: int = 790):
        self.stopping_rule_func = StoppingRule.fixed_to_k
        self.conditioning_rules = conditioning_rules
        self.return_periods_for_threshold = return_periods_for_threshold
        super(SimulationWithStoppingRuleAndConditioningForConditionedObservations, self).__init__(reference,
                                                                        historical_sample,
                                                                        n_iter,
                                                                        return_period,
                                                                        sample_size)
        rl_estimates = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        CI = {name: {x: [] for x in return_periods_for_threshold} for name, v in conditioning_rules}
        if USE_POOL:
            pool = Pool(cpu_count())
            results = pool.map(self.__call__, self.return_periods_for_threshold)
            pool.close()
        else:
            results = [self(x) for x in self.return_periods_for_threshold]
        for res in results:
            for name, v in res.items():
                for k, value in v.items():
                    rl_estimates[name][k] = list([p[0] for p in value])
                    CI[name][k] = list([p[1] for p in value])
        self.rl_estimates = rl_estimates
        self.CI = CI

    def __call__(self, x):
        time_start = time.time()
        res = {name: {x: []} for name, _ in self.conditioning_rules}
        for _ in range(self.n_iter):
            data = self.historical_sample
            sf_quantile = self.ref_distri.sf(x)
            f_quantile = self.ref_distri.cdf(x)
            u=Uniform()
            while len(data)< self.sample_size-1:
                random_below_thresh = [d for d in self.ref_distri.inverse_cdf(f_quantile*(u.rvs(self.sample_size-1))) if d < x]
                data = np.concatenate([data, random_below_thresh[:self.sample_size-len(data)-1]])
            data = pd.Series(np.concatenate([data, self.ref_distri.inverse_cdf(sf_quantile*u.rvs(1)+f_quantile)]))
            stopping_rule = StoppingRule(data, self.ref_distri,
                                         k=x, historical_sample_size=10,
                                         func=self.stopping_rule_func)
            data_stopped = stopping_rule.stopped_data()
            std_conditioning = ["Standard", "Excluding Last Obs"]
            crules = [(name, partial(crule, threshold=stopping_rule.threshold())) \
                      for (name, crule) in self.conditioning_rules if name
                      not in std_conditioning] \
                     + [(name, crule) for (name, crule) in self.conditioning_rules if name in std_conditioning]
            f = partial(to_run_in_parallel,
                        *[data_stopped, self.ref_distri, self.return_period])
            for name, c in crules:
                res[name][x].append(f((name, c)))
        time_elapsed = time.time() - time_start
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))
        return res

def to_run_in_parallel(data: pd.Series,
                       distribution: Distribution,
                       return_period: int,
                       conditioning_rule: Tuple[str, Callable]):
    name, cr = conditioning_rule
    try:
        likelihood = Likelihood(data=data,
                                distribution=distribution,
                                name=name,
                                conditioning_method=cr,
                                inference_confidence=0.95)
        rle = likelihood.return_level(return_period)
        rci = likelihood.return_level_confidence_interval(return_period)
    except:
        rle = distribution.isf(1/return_period)
        rci = [None, None]
    return rle, rci
