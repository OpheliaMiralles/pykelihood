import datetime
import io
import math
import sys
from abc import abstractmethod
from collections.abc import MutableSequence
from functools import partial, wraps
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple, Union

import cachetools
import numpy as np
import pandas as pd
import scipy.special
from rpy2 import robjects
from rpy2.robjects import conversion, pandas2ri
from rpy2.robjects.packages import importr
from scipy.optimize import minimize
from scipy.stats import beta, expon, genextreme, genpareto, norm, pareto, uniform

from pykelihood.parameters import ConstantParameter, Parameter, Parametrized
from pykelihood.stats_utils import ConditioningMethod

evd = importr('evd')
evmix = importr('evmix')
nm = importr("neldermead")
base = importr("base")
# Import Functions
fgev, qgev, pgev, dgev, rgev, profile_evd, clusters = evd.fgev, evd.qgev, evd.pgev, evd.dgev, evd.rgev, evd.profile_evd, evd.clusters
fpot = evd.fpot
qgpd, pgpd, dgpd, rgpd, fgpd = evd.qgpd, evd.pgpd, evd.dgpd, evd.rgpd, evmix.fgpd
fminsearch, nm_get, optimset = nm.fminsearch, nm.neldermead_get, nm.optimset
sum = base.sum
pandas2ri.activate()

EULER = -scipy.special.psi(1)


def hash_with_series(*args, **kwargs):
    to_hash = []
    for v in args:
        if isinstance(v, pd.Series):
            v = tuple(v.values)
        elif isinstance(v, MutableSequence) or isinstance(v, Iterable):
            v = tuple(v)
        to_hash.append(v)
    for k, v in kwargs.items():
        v = hash_with_series(v)
        to_hash.append((k, v))
    return hash(tuple(to_hash))


def ifnone(x, default):
    if x is None:
        return default
    return x


class Distribution:
    params: Tuple[Parameter]
    params_names: Tuple[str]
    flattened_params: Tuple[Parametrized]
    optimisation_params: Tuple[Parametrized]
    optimisation_param_dict: Dict[str, Parametrized]
    param_dict: Dict[str, Parametrized]

    def __hash__(self):
        return (self.__class__.__name__,) + self.params

    @abstractmethod
    def with_params(self, params: Iterable):
        return NotImplemented

    @abstractmethod
    def rvs(self, size: int):
        return NotImplemented

    @classmethod
    @abstractmethod
    def fit(cls, data: pd.Series, *args, **kwds):
        return NotImplemented

    @abstractmethod
    def cdf(self, x: Union[np.array, float]):
        return NotImplemented

    @abstractmethod
    def isf(self, q: float):
        return NotImplemented

    @abstractmethod
    def pdf(self, x: Union[np.array, float]):
        return NotImplemented

    @classmethod
    def param_dict_to_vec(cls, x: dict):
        return tuple(x.get(p) for p in cls.params_names)

    def sf(self, x: Union[np.array, float]):
        return 1 - self.cdf(x)

    def logcdf(self, x: Union[np.array, float]):
        return np.log(self.cdf(x))

    def logsf(self, x: Union[np.array, float]):
        return np.log(self.sf(x))

    def logpdf(self, x: Union[pd.Series, np.array, float], *args, **kwds):
        return np.log(self.pdf(x, *args, **kwds))

    def inverse_cdf(self, q: float):
        return self.isf(1 - q)

    def log_likelihood(self, data: Union[np.array, pd.Series],
                       conditioning_method: Callable = ConditioningMethod.no_conditioning,
                       *args, **kwds):
        res = self.logpdf(data, *args, **kwds)
        return np.sum(res - conditioning_method(data, self))

    def opposite_log_likelihood(self, data: Union[np.array, pd.Series],
                                conditioning_method: Callable = ConditioningMethod.no_conditioning,
                                *args, **kwds):
        return -self.log_likelihood(data, conditioning_method, *args, **kwds)

    def _process_fit_params(self, **kwds):
        param_dict = self.param_dict.copy()
        for name in param_dict:
            for name_fixed_param, value_fixed_param in kwds.items():
                if name_fixed_param == name:
                    param_dict[name] = ConstantParameter(value_fixed_param)
                elif name_fixed_param.startswith(name):
                    subparam = name_fixed_param.replace(f"{name}_", '')
                    subdic = param_dict[name].param_dict.copy()
                    subdic[subparam] = ConstantParameter(value_fixed_param)
                    param_dict[name] = param_dict[name].with_params(subdic.values())
        return param_dict

    def profile_likelihood(self, data,
                           conditioning_method: Callable = ConditioningMethod.no_conditioning,
                           fixed_params=None,
                           **kwds):
        param_dict = self._process_fit_params(**(fixed_params or {}))
        kwds.update(param_dict)
        return self.fit(data, conditioning_method=conditioning_method,
                        **kwds)


class AvoidAbstractMixin(object):
    def __getattribute__(self, item):
        x = object.__getattribute__(self, item)
        if hasattr(x, '__isabstractmethod__') and x.__isabstractmethod__ and hasattr(self, '__getattr__'):
            x = self.__getattr__(item)
        return x


class ScipyDistribution(Parametrized, Distribution, AvoidAbstractMixin):
    base_module: Any

    def _correct_trends(self, f):
        @wraps(f)
        def g(*args, **kwargs):
            res = f(*args, **kwargs)
            res = np.array(res)
            # if res is a matrix, then we only want the diagonal
            try:
                x, y = res.shape
            except ValueError:
                # only 1 dimension
                return res
            else:
                if x == 1 or y == 1:
                    return res.flatten()
                if x != y:
                    raise ValueError("Unexpected size of arguments")
                return np.diag(res)
        return g

    def __getattr__(self, item):
        if item not in ('pdf', 'logpdf', 'cdf', 'logcdf', 'ppf', 'isf', 'sf', 'logsf'):
            return super(ScipyDistribution, self).__getattr__(item)
        f = getattr(self.base_module, item)
        g = partial(self._wrapper, f)
        g = self._correct_trends(g)
        g = cachetools.cached(self._cache, key=hash_with_series)(g)
        self.__dict__[item] = g
        return g

    @classmethod
    def fit(cls, data, x0=None, conditioning_method=ConditioningMethod.no_conditioning, **fixed_values):
        init_parms = {}
        for k in cls.params_names:
            if k in fixed_values:
                v = fixed_values[k]
                if isinstance(v, Parametrized):
                    init_parms[k] = v
                else:
                    init_parms[k] = ConstantParameter(v)
        init = cls(**init_parms)
        x0 = x0 if x0 is not None else init.optimisation_params
        if len(x0) != len(init.optimisation_params):
            raise ValueError(f"Expected {len(init.params)} values in x0, got {len(x0)}")

        def to_minimize(x):
            o = init.with_params(x)
            return o.opposite_log_likelihood(data, conditioning_method=conditioning_method)

        res = minimize(to_minimize, x0, method="Nelder-Mead")
        return init.with_params(res.x)


class RDistribution(Parametrized, Distribution, AvoidAbstractMixin):
    def log_likelihood(self, data: Union[np.array, pd.Series, robjects.FloatVector],
                       conditioning_method: Callable = ConditioningMethod.no_conditioning,
                       *args, **kwds):
        res = self.logpdf(data, *args, **kwds)
        if hasattr(data, "rclass"):
            result = sum(res.ro - conditioning_method(data, self))
        else:
            result = sum(res.ro - conditioning_method(conversion.py2ri(data), self))
        return result

    def opposite_log_likelihood(self, data: Union[np.array, pd.Series],
                                conditioning_method: Callable = ConditioningMethod.no_conditioning,
                                *args, **kwds):
        return -self.log_likelihood(data, conditioning_method, *args, **kwds).ro

    @classmethod
    def shuffle(cls, fixed_param_name: Union[str, Sequence[str]]):
        if isinstance(fixed_param_name, str):
            fixed_param_name = fixed_param_name,
        return [x for x in cls.params_names if x in fixed_param_name] \
               + [x for x in cls.params_names if x not in fixed_param_name]

    @classmethod
    def fit(cls, data, x0=None, conditioning_method=ConditioningMethod.no_conditioning, **fixed_values):
        init_parms = {}
        for k in cls.params_names:
            if k in fixed_values:
                v = fixed_values[k]
                if isinstance(v, Parametrized):
                    init_parms[k] = v
                else:
                    init_parms[k] = ConstantParameter(v)
        init = cls(**init_parms)
        x0 = x0 if x0 is not None else init.optimisation_params
        if len(x0) != len(init.optimisation_params):
            raise ValueError(f"Expected {len(init.optimisation_params)} values in x0, got {len(x0)}")

        def to_minimize(x):
            o = init.with_params(x)
            return o.opposite_log_likelihood(data, conditioning_method=conditioning_method)

        try:
            data = conversion.py2ri(data)
            results = list(nm_get(fminsearch(to_minimize,
                                             x0=np.array(x0)), "xopt"))
        except:
            data = conversion.ri2py(data)
            results = minimize(to_minimize, np.array(x0),
                               bounds=[(None, None), (0, None), (None, None)],
                               method="Nelder-Mead").x
        return init.with_params(results)


class Uniform(ScipyDistribution):
    params_names = ("loc", "scale")
    base_module = uniform

    def __init__(self, loc=0., scale=1.):
        super(Uniform, self).__init__(loc, scale)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None):
        return f(x, ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None):
        return uniform.rvs(ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class Exponential(ScipyDistribution):
    params_names = ("loc", "rate")
    base_module = expon

    def __init__(self, loc=0., rate=1.):
        super(Exponential, self).__init__(loc, rate)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, rate=None):
        if rate is not None:
            rate = 1 / rate
        return f(x, ifnone(loc, self.loc()), ifnone(rate, 1 / self.rate()))

    def rvs(self, size, loc=None, rate=None):
        return self.base_module.rvs(ifnone(loc, self.loc()), ifnone(rate, 1 / self.rate()), size)


class Pareto(ScipyDistribution):
    params_names = ("loc", "scale", "alpha")
    base_module = pareto

    def __init__(self, loc=0., scale=1., alpha=1.):
        super(Pareto, self).__init__(loc, scale, alpha)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, alpha=None):
        return f(x, ifnone(alpha, self.alpha()), ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None, alpha=None):
        return self.base_module.rvs(ifnone(alpha, self.alpha()), ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class Beta(ScipyDistribution):
    params_names = ("loc", "scale", "alpha", "beta")
    base_module = beta

    def __init__(self, loc=0., scale=1., alpha=2., beta=1.):
        super(Beta, self).__init__(loc, scale, alpha, beta)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, alpha=None, beta=None):
        return f(x, ifnone(alpha, self.alpha()), ifnone(beta, self.beta()),
                 ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None, alpha=None, beta=None):
        return self.base_module.rvs(ifnone(alpha, self.alpha()), ifnone(beta, self.beta()),
                                    ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class Normal(ScipyDistribution):
    params_names = ("loc", "scale")
    base_module = norm

    def __init__(self, loc=0., scale=1.):
        super(Normal, self).__init__(loc, scale)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None):
        return f(x, ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None):
        return self.base_module.rvs(ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class GEV(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = genextreme

    def __init__(self, loc=0., scale=1., shape=0.):
        super(GEV, self).__init__(loc, scale, shape)
        self._cache = {}

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

    def _wrapper(self, f, x, loc=None, scale=None, shape=None):
        if shape is not None:
            shape = -shape
        return f(x, ifnone(shape, -self.shape()), ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, c=None, loc=None, scale=None):
        return self.base_module.rvs(ifnone(c, self.c), ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class GPD(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = genpareto

    def __init__(self, loc=0., scale=1., shape=0.):
        super(GPD, self).__init__(loc, scale, shape)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, shape=None):
        return f(x, ifnone(shape, self.shape()), ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None, shape=None):
        return self.base_module.rvs(ifnone(shape, self.shape()), ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class RGEV(RDistribution):
    params_names = ("loc", "scale", "shape")

    def __init__(self, loc=0., scale=1., shape=0.):
        super(RGEV, self).__init__(loc, scale, shape)

    @classmethod
    def fast_profile_likelihood(cls, data, name_of_fixed_parameter=None, conf=0.95):
        fitted = fgev(data.values, std_err=True)
        oldstdout = sys.stdout
        sys.stdout = io.StringIO()
        if name_of_fixed_parameter is not None:
            profile = pd.DataFrame(np.array(profile_evd(fitted, which=name_of_fixed_parameter, conf=conf))[0])[
                [0, 2, 3]] \
                .rename(columns={k: v for (k, v) in zip([0, 2, 3], cls.shuffle(name_of_fixed_parameter))})
            sys.stdout = oldstdout
            return profile
        else:
            profile = dict(profile_evd(fitted, conf=conf).items())
            profiles = {k: pd.DataFrame(np.array(profile[k]))[[0, 2, 3]] \
                .rename(columns={k: v for (k, v) in zip([0, 2, 3], cls.shuffle(k))}) for k in profile}
            sys.stdout = oldstdout
            return profiles

    def rvs(self, size: int,
            loc=None, scale=None, shape=None):
        return np.array(list(rgev(size, ifnone(loc, self.loc()),
                                  ifnone(scale, self.scale()),
                                  ifnone(shape, self.shape()))))

    def cdf(self, x: float,
            loc=None, scale=None, shape=None):
        if isinstance(x, Iterable):
            x = np.array(x)
        elif type(x) is int:
            x = float(x)
        result = pgev(x, ifnone(loc, self.loc()),
                      ifnone(scale, self.scale()),
                      ifnone(shape, self.shape()))
        return result

    def isf(self, q: float,
            loc=None, scale=None, shape=None):
        if isinstance(q, Iterable):
            q = np.array(q)
        elif type(q) is int:
            q = float(q)
        result = qgev(1 - q, ifnone(loc, self.loc()),
                      ifnone(scale, self.scale()),
                      ifnone(shape, self.shape()))
        return result

    def pdf(self, x: float,
            loc=None, scale=None, shape=None):
        if isinstance(x, Iterable):
            x = np.array(x)
        elif type(x) is int:
            x = float(x)
        result = dgev(x, ifnone(loc, self.loc()),
                      ifnone(scale, self.scale()),
                      ifnone(shape, self.shape()))
        return result

    def logpdf(self, x: float,
               loc=None, scale=None, shape=None):
        if isinstance(x, Iterable):
            x = np.array(x)
        elif type(x) is int:
            x = float(x)
        result = dgev(x, ifnone(loc, self.loc()),
                      ifnone(scale, self.scale()),
                      ifnone(shape, self.shape()), log=True)
        return result


class RGPD(RDistribution):
    params_names = ("loc", "scale", "shape")

    def __init__(self, loc=0.,
                 scale=1.,
                 shape=0.):
        super(RGPD, self).__init__(loc, scale, shape)

    def rvs(self, size: int, loc=None, scale=None, shape=None):
        return np.array(list(rgpd(size, ifnone(loc, self.loc()),
                                  ifnone(scale, self.scale()), ifnone(shape, self.shape()))))

    def cdf(self, x: float, loc=None, scale=None, shape=None):
        if isinstance(x, Iterable):
            x = np.array(x)
        elif type(x) == int:
            x = float(x)
        result = pgpd(x, ifnone(loc, self.loc()), ifnone(scale, self.scale()), ifnone(shape, self.shape()))
        return result

    def isf(self, q: float, loc=None, scale=None, shape=None):
        if isinstance(q, Iterable):
            q = np.array(q)
        elif type(q) == int:
            q = float(q)
        result = qgpd(1 - q, ifnone(loc, self.loc()),
                      ifnone(scale, self.scale()), ifnone(shape, self.shape()))
        return result

    def pdf(self, x: float, loc=None, scale=None, shape=None):
        if isinstance(x, Iterable):
            x = np.array(x)
        elif type(x) == int:
            x = float(x)
        result = dgpd(x, ifnone(loc, self.loc()), ifnone(scale, self.scale()), ifnone(shape, self.shape()))
        return result

    def logpdf(self, x: float, loc=None, scale=None, shape=None):
        if isinstance(x, Iterable):
            x = np.array(x)
        elif type(x) == int:
            x = float(x)
        result = dgpd(x, ifnone(loc, self.loc()), ifnone(scale, self.scale()), ifnone(shape, self.shape()), log=True)
        return result


class PointProcess(Distribution, Parametrized):
    def __init__(self, threshold: int,
                 jumps_size_distribution: Distribution = GPD(),
                 iat_distribution: Distribution = Exponential(),
                 remaining_points_distribution: Distribution = Uniform(ConstantParameter(0.), ConstantParameter(1.))):
        """

        :param threshold: defines what is a jump in term of exceeding of a certain threshold
        :param jumps_size_distribution: the jump size distribution
        :param iat_distribution: the inter-arrival times distribution with intensity lambda which can depend on time
        (non homogeneous Poisson Process) or history (Hawkes Process)
        :param remaining_points_distribution: optional, handles the points that are not jumps (appearing with frequency
        1-lambda on average)
        """
        super(PointProcess, self).__init__(*(iat_distribution.params +
                                             jumps_size_distribution.params +
                                             remaining_points_distribution.params))
        self.jumps_size = jumps_size_distribution
        self.iat = iat_distribution
        self.remaining_points = remaining_points_distribution
        self.threshold = threshold

    @property
    def params_names(self):
        IAT_names = [f"IAT_{p}" for p in self.iat.params_names]
        JS_names = [f"JS_{p}" for p in self.jumps_size.params_names]
        RemainingPoints_names = [f"RP_{p}" for p in self.remaining_points.params_names]
        return tuple(IAT_names) + tuple(JS_names) + tuple(RemainingPoints_names)

    def _process_fit_params(self, **kwds):
        IAT_kwds = {k.replace("IAT_", ""): v for (k,v) in kwds.items() if k.startswith("IAT_")}
        JS_kwds = {k.replace("JS_", ""): v for (k,v) in kwds.items() if k.startswith("JS_")}
        RP_kwds = {k.replace("RP_", ""): v for (k,v) in kwds.items() if k.startswith("RP_")}
        IAT_params = self.iat._process_fit_params(**IAT_kwds)
        JS_params = self.jumps_size._process_fit_params(**JS_kwds)
        RP_params = self.remaining_points._process_fit_params(**RP_kwds)
        if len(kwds) > len(IAT_kwds) + len(JS_kwds) + len(RP_kwds):
            raise ValueError(f"Unexpected parameters encountered in keywords: {kwds}")
        return IAT_params, JS_params, RP_params

    def profile_likelihood(self, data,
                           conditioning_method: Callable = ConditioningMethod.no_conditioning,
                           fixed_params=None,
                           **kwds):
        kwds.update(**(fixed_params or {}))
        return self.fit(data, conditioning_method=conditioning_method,
                        **kwds)

    def fit(self, data: pd.Series,
            conditioning_method=ConditioningMethod.no_conditioning,
            x0=None, opt_method="Nelder-Mead",
            *args, **kwds):
        params_iat, params_js, params_rp = self._process_fit_params(**kwds)
        obj = type(self)(type(self.iat)(**params_iat),
                         type(self.jumps_size)(**params_js),
                         type(self.remaining_points)(**params_rp))
        x0 = obj.optimisation_params if x0 is None else x0

        def to_minimize(var_params):
            return obj.with_params(var_params).opposite_log_likelihood(data, conditioning_method)

        result = minimize(to_minimize,
                          method=opt_method,
                          options={"adaptive": True, 'maxfev': 1000},
                          x0=x0)
        res = list(result.x)
        return obj.with_params(res)

    def rvs(self, size):
        freq = 1 / self.counting_process_distribution.scale
        threshold = self.threshold
        nb_exceedances = int(freq * size)
        expo = np.cumsum(self.counting_process_distribution.rvs(size=nb_exceedances))
        exceedances = np.array(self.jumps_size_distribution.rvs(nb_exceedances))
        non_exceedances = size - nb_exceedances
        others = threshold * self.under_threshold_distribution.rvs(size=non_exceedances)
        sizes = np.insert(others, [math.ceil(e) for e in expo], exceedances)
        return np.array(list(enumerate(sizes)))

    def cdf(self, x):
        freq = 1 / self.counting_process_distribution.scale
        res = np.where(x < self.threshold,
                       self.under_threshold_distribution.cdf(x) * (1 - freq),
                       freq * self.jumps_size_distribution.cdf(x))
        if isinstance(x, float):
            return res[0]
        else:
            return res

    def isf(self, q):
        freq = self.counting_process_distribution.scale
        res = np.where(q > freq,
                       self.under_threshold_distribution.isf(q * (1 - freq)),
                       self.jumps_size_distribution.isf(q * freq))
        return res

    def pdf(self, x):
        freq = self.counting_process_distribution.scale
        res = np.where(x < self.threshold, (1 - freq) * self.under_threshold_distribution.pdf(x),
                       self.jumps_size_distribution.pdf(x) * freq)
        return res


class CompositionDistribution(Distribution, Parametrized):
    def __init__(self, d1: Distribution, d2: Distribution):
        """
        Distribution of cdf GoF
        :param d1: distribution aimed at replacing the uniform sampling in the formula used
        to generate variables with distribution d2 (G in F^{-1}(G^{-1}(U))
        :param d2: main objective distribution, especially aimed at determining the asymptotic behaviour.
        """
        super(CompositionDistribution, self).__init__(*(d1.params+d2.params))
        self.d1 = d1
        self.d2 = d2

    @property
    def params_names(self):
        d1_names = [f"d1_{p}" for p in self.d1.params_names]
        d2_names = [f"d2_{p}" for p in self.d2.params_names]
        return tuple(d1_names) + tuple(d2_names)

    def with_params(self, new_params):
        new_params = iter(new_params)
        d1 = self.d1.with_params(new_params)
        d2 = self.d2.with_params(new_params)
        return type(self)(d1, d2)

    def rvs(self, size):
        return self.d2.inverse_cdf(self.d1.rvs(size))

    def cdf(self, x):
        return self.d1.cdf(self.d2.cdf(x))

    def isf(self, q):
        return self.d2.isf(self.d1.isf(1 - q))

    def pdf(self, x, *args):
        return self.d2.pdf(x) * self.d1.pdf(self.d2.cdf(x))

    def _process_fit_params(self, **kwds):
        d1_kwds = {k.replace("d1_", ""): v for (k,v) in kwds.items() if k.startswith("d1_")}
        d2_kwds = {k.replace("d2_", ""): v for (k,v) in kwds.items() if k.startswith("d2_")}
        d1_params = self.d1._process_fit_params(**d1_kwds)
        d2_params = self.d2._process_fit_params(**d2_kwds)
        if len(kwds) > len(d1_kwds) + len(d2_kwds) :
            raise ValueError(f"Unexpected parameters encountered in keywords: {kwds}")
        return d1_params, d2_params

    def profile_likelihood(self, data,
                           conditioning_method: Callable = ConditioningMethod.no_conditioning,
                           fixed_params=None,
                           **kwds):
        kwds.update(**(fixed_params or {}))
        return self.fit(data, conditioning_method=conditioning_method,
                        **kwds)

    def fit(self, data: pd.Series,
            conditioning_method=ConditioningMethod.no_conditioning,
            x0=None, opt_method="Nelder-Mead",
            *args, **kwds):
        params_d1, params_d2 = self._process_fit_params(**kwds)
        obj = type(self)(type(self.d1)(**params_d1), type(self.d2)(**params_d2))
        x0 = obj.optimisation_params if x0 is None else x0

        def to_minimize(var_params):
            return obj.with_params(var_params).opposite_log_likelihood(data, conditioning_method)

        result = minimize(to_minimize,
                          method=opt_method,
                          options={"adaptive": True, 'maxfev': 1000},
                          x0=x0)
        res = list(result.x)
        return obj.with_params(res)
