import datetime
import io
import math
import sys
from abc import abstractmethod
from collections.abc import MutableSequence
from functools import partial
from typing import Sequence, Iterable, Union, Callable, Tuple, Any

import cachetools
import numpy as np
import pandas as pd
import scipy.special
from rpy2 import robjects
from rpy2.robjects import pandas2ri, conversion
from rpy2.robjects.packages import importr
from scipy.optimize import minimize
from scipy.stats import uniform, pareto, beta, expon, norm, genextreme, genpareto

from pykelihood.parameters import Parameter, Parametrized, ConstantParameter
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
    _params: Tuple[Parameter]
    params_names: Tuple[str]

    def __hash__(self):
        return (self.__class__.__name__,) + self.params

    @abstractmethod
    def params(self):
        return NotImplemented

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

    def profile_likelihood(self, data, name_fixed_param, value_fixed_param,
                           conditioning_method: Callable = ConditioningMethod.no_conditioning, **kwds):
        kwds.update({name_fixed_param: value_fixed_param})
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

    def __getattr__(self, item):
        if item not in ('pdf', 'logpdf', 'cdf', 'logcdf', 'ppf', 'isf', 'sf', 'logsf'):
            return super(ScipyDistribution, self).__getattr__(item)
        f = getattr(self.base_module, item)
        g = partial(self._wrapper, f)
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
        x0 = x0 if x0 is not None else init.params
        if len(x0) != len(init.params):
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
            return result.ro
        else:
            return np.sum(res - conditioning_method(data, self))

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
        x0 = x0 if x0 is not None else init.params
        if len(x0) != len(init.params):
            raise ValueError(f"Expected {len(init.params)} values in x0, got {len(x0)}")

        def to_minimize(x):
            o = init.with_params(x)
            return o.opposite_log_likelihood(data, conditioning_method=conditioning_method)

        try:
            data = conversion.py2ri(data)
            results = list(nm_get(fminsearch(to_minimize,
                                             x0=np.array(x0)), "xopt"))
        except:
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
        return uniform.rvs(ifnone(loc, self.loc()), ifnone(rate, 1 / self.scale()), size)


class Pareto(ScipyDistribution):
    params_names = ("loc", "scale", "alpha")
    base_module = pareto

    def __init__(self, loc=0., scale=1., alpha=1.):
        super(Pareto, self).__init__(loc, scale, alpha)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, alpha=None):
        return f(x, ifnone(alpha, self.alpha()), ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None, alpha=None):
        return pareto.rvs(ifnone(alpha, self.alpha()), ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


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
        return beta.rvs(ifnone(alpha, self.alpha()), ifnone(beta, self.beta()),
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
        return norm.rvs(ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


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
        return genextreme.rvs(ifnone(c, self.c), ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class GPD(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = genpareto

    def __init__(self, loc=0., scale=1., shape=0.):
        super(GPD, self).__init__(loc, scale, shape)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, shape=None):
        return f(x, ifnone(shape, self.shape()), ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None, shape=None):
        return genpareto.rvs(ifnone(shape, self.shape()), ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


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


class PointProcess(Distribution):
    def __init__(self, threshold: int,
                 pot_distribution: Distribution = GPD,
                 iat_distribution: Distribution = Exponential,
                 under_threshold_distribution=Uniform,
                 npp=365.25):
        self.pot_distribution = pot_distribution
        self.iat_distribution = iat_distribution
        self.under_threshold_distribution = under_threshold_distribution
        self.threshold = threshold
        self.npp = npp

    @property
    def params(self):
        return self.pot_distribution.params + self.iat_distribution.params

    @property
    def params_names(self):
        return self.pot_distribution.params_names + self.iat_distribution.params_names

    def fit(self, data: pd.Series,
            conditioning_method=ConditioningMethod.no_conditioning, *args, **kwds):
        """
        :param data:
        :param args: model: gpd or pp representation;
         threshold very useful for POT;
         npp is the number of obs per period (ex: daily observations that we want to
         put in yearly units -> 365.25 obs/period (default)
        :param kwds:
        :return:
        """
        pot = clusters(np.array(data), u=self.threshold, cmax=True)
        pot_dist = self.pot_distribution.fit(pot, conditioning_method=conditioning_method, *args, **kwds)
        if isinstance(list(data.index.values)[0], datetime.date):
            iat = pot.reset_index().diff()[data.index.name].dropna().apply(lambda x: x.days)
        else:
            iat = np.diff(np.where(data >= self.threshold)[0])
        iat = iat / len(data)
        iat_dist = self.iat_distribution.fit(iat, floc=0.)
        ut_dist = self.under_threshold_distribution.fit(data[data < self.threshold], floc=0.)
        return PointProcess(self.threshold, pot_dist, iat_dist, ut_dist)

    def rvs(self, size):
        freq = 1 / self.iat_distribution.scale
        threshold = self.threshold
        nb_exceedances = int(freq * size)
        expo = np.cumsum(self.iat_distribution.rvs(size=nb_exceedances))
        exceedances = np.array(self.pot_distribution.rvs(nb_exceedances))
        non_exceedances = size - nb_exceedances
        others = threshold * self.under_threshold_distribution.rvs(size=non_exceedances)
        sizes = np.insert(others, [math.ceil(e) for e in expo], exceedances)
        return np.array(list(enumerate(sizes)))

    def cdf(self, x):
        freq = 1 / self.iat_distribution.scale
        res = np.where(x < self.threshold,
                       self.under_threshold_distribution.cdf(x) * (1 - freq),
                       freq * self.pot_distribution.cdf(x))
        if isinstance(x, float):
            return res[0]
        else:
            return res

    def isf(self, q):
        freq = self.iat_distribution.scale
        res = np.where(q > freq,
                       self.under_threshold_distribution.isf(q * (1 - freq)),
                       self.pot_distribution.isf(q * freq))
        return res

    def pdf(self, x):
        freq = self.iat_distribution.scale
        res = np.where(x < self.threshold, (1 - freq) * self.under_threshold_distribution.pdf(x),
                       self.pot_distribution.pdf(x) * freq)
        return res


class JointDistribution(Distribution):
    def __init__(self, d1: Distribution, d2: Distribution):
        self.d1 = d1
        self.d2 = d2

    @property
    def params_names(self):
        d1_names = [f"d1_{p}" for p in self.d1.params_names]
        d2_names = [f"d2_{p}" for p in self.d2.params_names]
        return tuple(d1_names) + tuple(d2_names)

    @property
    def params(self):
        return self.d1.params + self.d2.params

    @property
    def _params(self):
        return self.d1._params + self.d2._params

    @property
    def names_and_params(self):
        for name, p in self.d1.names_and_params:
            yield f"d1_{name}", p
        for name, p in self.d2.names_and_params:
            yield f"d2_{name}", p

    def with_params(self, new_params):
        new_params = iter(new_params)
        d1 = self.d1.with_params(new_params)
        d2 = self.d2.with_params(new_params)
        return type(self)(d1, d2)

    def rvs(self, size):
        return self.d2.inverse_cdf(self.d1.inverse_cdf(Uniform().rvs(size)))

    def cdf(self, x):
        return self.d1.cdf(self.d2.cdf(x))

    def isf(self, q):
        return self.d2.isf(self.d1.isf(q))

    def pdf(self, x, *args):
        return self.d2.pdf(x) * self.d1.pdf(self.d2.cdf(x))

    def _process_fit_params(self, **kwds):
        d1_params = []
        for name, _ in self.d1.names_and_params:
            k = f"d1_{name}"
            if k in kwds:
                d1_params.append(ConstantParameter(kwds.pop(k)))
            else:
                d1_params.append(getattr(self.d1, name))
        d2_params = []
        for name, _ in self.d2.names_and_params:
            k = f"d2_{name}"
            if k in kwds:
                d2_params.append(ConstantParameter(kwds.pop(k)))
            else:
                d2_params.append(getattr(self.d2, name))
        if kwds:
            raise ValueError(f"Unexpected parameters: {kwds}")
        return d1_params, d2_params

    def fit(self, data: pd.Series,
            conditioning_method=ConditioningMethod.no_conditioning,
            x0=None, opt_method="Nelder-Mead",
            *args, **kwds):
        params_d1, params_d2 = self._process_fit_params(**kwds)
        obj = self.with_params(params_d1 + params_d2)

        x0 = obj.params if x0 is None else x0

        def to_minimize(var_params):
            return obj.with_params(var_params).opposite_log_likelihood(data, conditioning_method)

        result = minimize(to_minimize,
                          method=opt_method,
                          options={"adaptive": True, 'maxfev': 1000},
                          x0=x0)
        res = list(result.x)
        return obj.with_params(res)
