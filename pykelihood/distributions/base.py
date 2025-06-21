from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import OptimizeResult, minimize

from pykelihood.generic_types import Obs
from pykelihood.metrics import opposite_log_likelihood
from pykelihood.parameters import Parametrized, ensure_parametrized

if TYPE_CHECKING:
    from typing import Self

_T = TypeVar("_T")
SomeDistribution = TypeVar("SomeDistribution", bound="Distribution")


class Distribution(Parametrized, ABC):
    """
    Base class for all distributions.

    Methods
    -------
    rvs(size: int, *args, **kwargs) -> np.ndarray
        Generate random variates.
    cdf(x: Obs)
        Cumulative distribution function.
    isf(q: Obs)
        Inverse survival function.
    ppf(q: Obs)
        Percent point function (inverse of cdf).
    pdf(x: Obs)
        Probability density function.
    sf(x: Obs)
        Survival function.
    logcdf(x: Obs)
        Log of the cumulative distribution function.
    logsf(x: Obs)
        Log of the survival function.
    logpdf(x: Obs)
        Log of the probability density function.
    inverse_cdf(q: Obs)
        Inverse of the cumulative distribution function.
    fit(data: Obs, x0: Sequence[float] = None, score: Callable[["Distribution", Obs], float] = opposite_log_likelihood, scipy_args: Optional[Dict] = None, **fixed_values) -> SomeDistribution
        Fit the distribution to the data.
    """

    def __hash__(self):
        return (self.__class__.__name__,) + self.params

    @abstractmethod
    def rvs(self, size: int, *args, **kwargs) -> np.ndarray:
        return NotImplemented

    @abstractmethod
    def cdf(self, x: Obs):
        return NotImplemented

    @abstractmethod
    def isf(self, q: Obs):
        return NotImplemented

    @abstractmethod
    def ppf(self, q: Obs):
        return NotImplemented

    @abstractmethod
    def pdf(self, x: Obs):
        return NotImplemented

    @classmethod
    def param_dict_to_vec(cls, x: dict):
        return tuple(x.get(p) for p in cls.params_names)

    def sf(self, x: Obs):
        return 1 - self.cdf(x)

    def logcdf(self, x: Obs):
        return np.log(self.cdf(x))

    def logsf(self, x: Obs):
        return np.log(self.sf(x))

    def logpdf(self, x: Obs):
        return np.log(self.pdf(x))

    def inverse_cdf(self, q: Obs):
        if hasattr(self, "ppf"):
            return self.ppf(q)
        else:
            return self.isf(1 - q)

    def _apply_constraints(self, data):
        return data

    def fit(
        self,
        data: Obs,
        x0: Sequence[float] | None = None,
        score: Callable[[Distribution, Obs], float] = opposite_log_likelihood,
        scipy_args: dict | None = None,
        **fixed_values,
    ) -> Fit[Self]:
        """
        Fit the distribution to the data.

        Parameters
        ----------
        data : Obs
            Data to fit the distribution to.
        x0 : Sequence[float], optional
            Initial guess for the parameters, by default None.
        score : Callable[["Distribution", Obs], float], optional
            Scoring function, by default opposite_log_likelihood.
        scipy_args : Optional[Dict], optional
            Additional arguments for scipy.optimize.minimize, by default None.
        fixed_values : dict
            Fixed values for the parameters.

        Returns
        -------
        The result of the fit. A new instance is created with the fitted parameters.
        """
        init_parms = self._process_fit_params(**fixed_values)
        init = type(self)(**init_parms)
        data = init._apply_constraints(data)

        if x0 is None:
            x0 = [x() for x in init.optimisation_params]
        else:
            if len(x0) != len(init.optimisation_params):
                raise ValueError(
                    f"Expected {len(init.optimisation_params)} values in x0, got {len(x0)}"
                )
            x0 = [float(x) for x in x0]

        def to_minimize(x) -> float:
            return score(init.with_params(x), data)

        minimize_args = {
            "method": "Nelder-Mead",
            "options": {"maxiter": 1500, "fatol": 1e-8},
        }
        minimize_args.update(scipy_args or {})
        optimization_result = minimize(to_minimize, x0, **minimize_args)
        dist = init.with_params(optimization_result.x)

        return Fit(dist, data, score, x0=x0, optimize_result=optimization_result)

    def fit_instance(self, *args, **kwargs):
        warnings.warn(
            "fit_instance is deprecated, use fit instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fit(*args, **kwargs)

    def _process_fit_params(self, **kwds):
        out_dict = self.param_dict.copy()
        to_remove = set()
        all_fixed_params = {}
        for name_fixed_param, value_fixed_param in kwds.items():
            for _, related_params in self.param_mapping():
                if name_fixed_param in related_params and len(related_params) >= 1:
                    for p in related_params:
                        all_fixed_params[p] = value_fixed_param
        for param_name in out_dict:
            for name_fixed_param, value_fixed_param in all_fixed_params.items():
                if name_fixed_param.startswith(param_name):
                    # this name is being processed, no need to keep it
                    to_remove.add(name_fixed_param)
                    replacement = ensure_parametrized(value_fixed_param, constant=True)
                    if name_fixed_param == param_name:
                        out_dict[param_name] = replacement
                    else:
                        sub_name = name_fixed_param.replace(f"{param_name}_", "")
                        old_param = out_dict[param_name]
                        new_param = old_param.with_params(**{sub_name: replacement})
                        out_dict[param_name] = new_param
        # other non-parameter kw arguments
        for name, value in kwds.items():
            if name not in to_remove:
                out_dict[name] = value
        return out_dict


@dataclass
class Fit(Generic[_T]):
    fitted: _T
    data: Obs
    score_fn: Callable[[_T, Obs], float]
    x0: Sequence[float]
    optimize_result: OptimizeResult

    def confidence_interval(
        self, param: str, alpha: float = 0.05, precision: float = 1e-5
    ) -> tuple[float, float]:
        """
        Calculate the confidence interval for a parameter.

        Parameters
        ----------
        param : str
            Name of the parameter.
        alpha : float, optional
            Significance level, by default 0.05.

        Returns
        -------
        tuple
            Lower and upper bounds of the confidence interval.
        """
        if param not in self.fitted.params_names:
            raise ValueError(f"Parameter {param} not found in fitted distribution.")

        from pykelihood.profiler import Profiler

        profiler = Profiler(
            self.fitted,
            self.data,
            self.score_fn,
            single_profiling_param=param,
            inference_confidence=alpha,
        )
        return profiler.confidence_interval(param, precision=precision)

    # TODO: implement explicit wrappers and use this only for dynamic attributes (e.g. param names)
    def __getattr__(self, item: str):
        return getattr(self.fitted, item)


class ScipyDistribution(Distribution, ABC):
    """
    Base class for distributions based on SciPy.
    """

    _base_module: stats.rv_continuous

    @abstractmethod
    def _to_scipy_args(self, **params) -> dict[str, ArrayLike]:
        raise NotImplementedError

    def rvs(self, size=None, random_state=None, **kwargs):
        return self._base_module.rvs(
            **self._to_scipy_args(**kwargs), size=size, random_state=random_state
        )

    def pdf(self, x, **kwargs):
        return self._base_module.pdf(x, **self._to_scipy_args(**kwargs))

    def cdf(self, x, **kwargs):
        return self._base_module.cdf(x, **self._to_scipy_args(**kwargs))

    def isf(self, q, **kwargs):
        return self._base_module.isf(q, **self._to_scipy_args(**kwargs))

    def ppf(self, q, **kwargs):
        return self._base_module.ppf(q, **self._to_scipy_args(**kwargs))

    def sf(self, x, **kwargs):
        return self._base_module.sf(x, **self._to_scipy_args(**kwargs))

    def logpdf(self, x, **kwargs):
        return self._base_module.logpdf(x, **self._to_scipy_args(**kwargs))

    def logcdf(self, x, **kwargs):
        return self._base_module.logcdf(x, **self._to_scipy_args(**kwargs))

    def logsf(self, x, **kwargs):
        return self._base_module.logsf(x, **self._to_scipy_args(**kwargs))
