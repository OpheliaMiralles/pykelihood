from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Sequence, Type, TypeVar

import numpy as np
import scipy.special
from scipy.optimize import minimize
from scipy.stats import (
    beta,
    expon,
    gamma,
    genextreme,
    genpareto,
    lognorm,
    norm,
    pareto,
    uniform,
)

from pykelihood.generic_types import Obs
from pykelihood.metrics import opposite_log_likelihood
from pykelihood.parameters import ConstantParameter, Parametrized, ensure_parametrized
from pykelihood.utils import ifnone

T = TypeVar("T")
SomeDistribution = TypeVar("SomeDistribution", bound="Distribution")

EULER = -scipy.special.psi(1)


class Distribution(Parametrized):
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

    @classmethod
    def fit(
        cls: Type[SomeDistribution],
        data: Obs,
        x0: Sequence[float] = None,
        score: Callable[["Distribution", Obs], float] = opposite_log_likelihood,
        **fixed_values,
    ) -> SomeDistribution:
        init_parms = {}
        for k in cls.params_names:
            if k in fixed_values:
                v = fixed_values.pop(k)
                if isinstance(v, Parametrized):
                    init_parms[k] = v
                else:
                    init_parms[k] = ConstantParameter(v)
        # Add keyword arguments useful for object creation
        for k, v in fixed_values.items():
            if k not in init_parms:
                init_parms[k] = v
        init = cls(**init_parms)
        x0 = x0 or [x.value for x in init.optimisation_params]
        if len(x0) != len(init.optimisation_params):
            raise ValueError(
                f"Expected {len(init.optimisation_params)} values in x0, got {len(x0)}"
            )
        data = init._apply_constraints(data)

        def to_minimize(x) -> float:
            return score(init.with_params(x), data)

        res = minimize(to_minimize, x0, method="Nelder-Mead", options={"maxiter": 1500})
        return init.with_params(res.x)

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

    def fit_instance(
        self,
        data,
        score=opposite_log_likelihood,
        x0: Sequence[float] = None,
        **fixed_values,
    ):
        param_dict = self._process_fit_params(**fixed_values)
        return self.fit(data, score=score, x0=x0, **param_dict)


class AvoidAbstractMixin(object):
    def __getattribute__(self, item):
        x = object.__getattribute__(self, item)
        if (
            hasattr(x, "__isabstractmethod__")
            and x.__isabstractmethod__
            and hasattr(self, "__getattr__")
        ):
            x = self.__getattr__(item)
        return x


class ScipyDistribution(Distribution, AvoidAbstractMixin):
    base_module: Any

    def rvs(self, size=None, random_state=None, **kwargs):
        base_rvs = getattr(self.base_module, "rvs")
        params = {p: kwargs.pop(p) for p in self.params_names if p in kwargs}
        return base_rvs(
            **self._to_scipy_args(**params),
            size=size,
            random_state=random_state,
            **kwargs,
        )

    def _wrapper(self, f, x, **extra_args):
        params = {}
        other_args = {}
        for key, value in extra_args.items():
            if key in self.params_names:
                params[key] = value
            else:
                other_args[key] = value
        return f(x, **self._to_scipy_args(**params), **other_args)

    def __getattr__(self, item):
        if item not in (
            "pdf",
            "logpdf",
            "cdf",
            "logcdf",
            "ppf",
            "isf",
            "sf",
            "logsf",
        ):
            return super(ScipyDistribution, self).__getattr__(item)
        f = getattr(self.base_module, item)
        g = partial(self._wrapper, f)
        # g = _correct_trends(g)
        self.__dict__[item] = g
        return g


class Uniform(ScipyDistribution):
    params_names = ("loc", "scale")
    base_module = uniform

    def __init__(self, loc=0.0, scale=1.0):
        super(Uniform, self).__init__(loc, scale)

    def _to_scipy_args(self, loc=None, scale=None):
        return {"loc": ifnone(loc, self.loc()), "scale": ifnone(scale, self.scale())}


class Exponential(ScipyDistribution):
    params_names = ("loc", "rate")
    base_module = expon

    def __init__(self, loc=0.0, rate=1.0):
        super(Exponential, self).__init__(loc, rate)

    def _to_scipy_args(self, loc=None, rate=None):
        if rate is not None:
            rate = 1 / rate
        return {"loc": ifnone(loc, self.loc()), "scale": ifnone(rate, 1 / self.rate())}


class Gamma(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = gamma

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super(Gamma, self).__init__(loc, scale, shape)

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        return {
            "a": ifnone(shape, self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class Pareto(ScipyDistribution):
    params_names = ("loc", "scale", "alpha")
    base_module = pareto

    def __init__(self, loc=0.0, scale=1.0, alpha=1.0):
        super(Pareto, self).__init__(loc, scale, alpha)

    def _to_scipy_args(self, loc=None, scale=None, alpha=None):
        return {
            "c": ifnone(alpha, self.alpha()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class Beta(ScipyDistribution):
    params_names = ("loc", "scale", "alpha", "beta")
    base_module = beta

    def __init__(self, loc=0.0, scale=1.0, alpha=2.0, beta=1.0):
        super(Beta, self).__init__(loc, scale, alpha, beta)

    def _to_scipy_args(self, loc=None, scale=None, alpha=None, beta=None):
        return {
            "a": ifnone(alpha, self.alpha()),
            "b": ifnone(beta, self.beta()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class Normal(ScipyDistribution):
    params_names = ("loc", "scale")
    base_module = norm

    def __init__(self, loc=0.0, scale=1.0):
        super(Normal, self).__init__(loc, scale)

    def _to_scipy_args(self, loc=None, scale=None):
        return {"loc": ifnone(loc, self.loc()), "scale": ifnone(scale, self.scale())}


class LogNormale(ScipyDistribution):
    params_names = ("loc", "scale")
    base_module = lognorm

    def __init__(self, loc=0.0, scale=1.0):
        super(LogNormale, self).__init__(loc, scale)

    def _to_scipy_args(self, loc=None, scale=None):
        return {"s": ifnone(scale, self.scale()), "loc": ifnone(loc, self.loc())}


class GEV(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = genextreme

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super(GEV, self).__init__(loc, scale, shape)

    def lb_shape(self, data):
        x_min = data.min()
        x_max = data.max()
        if x_min * x_max < 0:
            return -np.inf
        elif x_min > 0:
            return self.scale / (x_max - self.loc())
        else:
            return self.scale / (x_min - self.loc())

    def ub_shape(self, data):
        x_min = data.min()
        x_max = data.max()
        if x_min * x_max < 0:
            return np.inf
        elif x_min > 0:
            return self.scale / (x_min - self.loc())
        else:
            return self.scale / (x_max - self.loc())

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        if shape is not None:
            shape = -shape
        return {
            "c": ifnone(shape, -self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class GPD(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = genpareto

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super(GPD, self).__init__(loc, scale, shape)

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        return {
            "c": ifnone(shape, self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class TruncatedDistribution(Distribution):
    params_names = ("distribution",)

    def __init__(
        self, distribution: Distribution, lower_bound=-np.inf, upper_bound=np.inf
    ):
        if upper_bound == lower_bound:
            raise ValueError("Both bounds are equal.")
        super(TruncatedDistribution, self).__init__(distribution)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        lower_cdf = self.distribution.cdf(self.lower_bound)
        upper_cdf = self.distribution.cdf(self.upper_bound)
        self._normalizer = upper_cdf - lower_cdf

    def _build_instance(self, **new_params):
        distribution = new_params.pop("distribution")
        if new_params:
            raise ValueError(f"Unexpected arguments: {new_params}")
        return type(self)(distribution, self.lower_bound, self.upper_bound)

    def _valid_indices(self, x: np.ndarray):
        return (self.lower_bound <= x) & (x <= self.upper_bound)

    def _apply_constraints(self, x):
        return x[self._valid_indices(x)]

    def fit_instance(self, *args, **kwargs):
        kwargs.update(lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return super().fit_instance(*args, **kwargs)

    def rvs(self, size: int, *args, **kwargs):
        u = Uniform(
            self.distribution.cdf(self.lower_bound),
            self.distribution.cdf(self.upper_bound),
        )
        return self.distribution.inverse_cdf(u.rvs(size, *args, **kwargs))

    def pdf(self, x: Obs):
        return np.where(
            self._valid_indices(x),
            self.distribution.pdf(x) / self._normalizer,
            0.0,
        )

    def cdf(self, x: Obs):
        right_range_x = (
            self.distribution.cdf(x) - self.distribution.cdf(self.lower_bound)
        ) / self._normalizer
        return np.where(self._valid_indices(x), right_range_x, 0.0)

    def isf(self, q: Obs):
        return self.distribution.isf(
            self.distribution.isf(self.upper_bound) + q * self._normalizer
        )

    def ppf(self, q: Obs):
        return self.distribution.ppf(
            self.distribution.cdf(self.lower_bound) + q * self._normalizer
        )
