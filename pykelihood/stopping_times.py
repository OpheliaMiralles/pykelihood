import warnings
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pykelihood.distributions import Distribution

warnings.filterwarnings('ignore')


class StoppingRule(object):
    def __init__(self, data: pd.Series,
                 distribution: "Distribution",
                 k: int,
                 historical_sample_size: int,
                 func: Callable[[pd.Series, int, "Distribution", int], int]):
        """

        :param data: database that is the base of the analysis once we stop
        :param historical_sample_size: number of data points necessary to provide a reliable first estimate
        :param func: stopping rule (it can be fixed or variable as the static methods detailed in the class)
        """
        self.data = data
        self.k = k
        self.distribution = distribution
        self.stopping_rule = func
        self.historical_sample_size = historical_sample_size

    def __call__(self):
        return self.stopping_rule(self.data, self.historical_sample_size,
                                  self.distribution, self.k)

    def stopped_data(self):
        c, N = self.__call__()
        return self.data.iloc[:N]

    def threshold(self):
        c, N = self.__call__()
        return c

    def last_index(self):
        c, N = self.__call__()
        return N

    @staticmethod
    def fixed_to_middle(data: pd.Series, historical_sample_size: int,
                        distribution: "Distribution", k: int):
        max = data.iloc[historical_sample_size:].max()
        min = data.iloc[historical_sample_size:].min()
        c = (max - min) / 2
        first_index_above_threshold = np.argmax(data.iloc[historical_sample_size:] >= c)
        if first_index_above_threshold == 0 and sum(data.iloc[historical_sample_size:] >= c) == 0:
            N = len(data)
        else:
            N = historical_sample_size + first_index_above_threshold + 1
        number_of_tests_threshold = N - historical_sample_size
        return [c] * number_of_tests_threshold, N

    @staticmethod
    def fixed_to_k(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        c = k
        first_index_above_threshold = np.argmax(data.iloc[historical_sample_size:] >= c)
        if first_index_above_threshold == 0 and sum(data.iloc[historical_sample_size:] >= c) == 0:
            N = len(data)
        else:
            N = historical_sample_size + first_index_above_threshold + 1
        number_of_tests_threshold = N - historical_sample_size
        return [c] * number_of_tests_threshold, N

    @staticmethod
    def variable(data: pd.Series, historical_sample_size: int, distribution: "Distribution", k: int):
        j = historical_sample_size
        fit = distribution.fit(data.iloc[:j])
        return_level_estimate = fit.isf(1 / k)
        return_level_estimates = [float(return_level_estimate)]
        data_stopped = data.iloc[:j + 1]
        j += 1
        while data_stopped.iloc[-1] < return_level_estimate and len(data_stopped) < len(data):
            fit = distribution.fit(data_stopped)
            return_level_estimate = fit.isf(1 / k)
            return_level_estimates.append(float(return_level_estimate))
            data_stopped = data.iloc[:j + 1]
            j += 1
        N = j
        return return_level_estimates, N
