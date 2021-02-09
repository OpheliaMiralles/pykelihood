from abc import abstractmethod
from collections.abc import MutableSequence
from functools import partial, wraps
from typing import Any, Callable, Iterable, Union

import cachetools
import numpy as np
import pandas as pd
import scipy.special
from scipy.optimize import minimize
from scipy.stats import beta, expon, gamma, genextreme, genpareto, norm, pareto, uniform

from pykelihood import kernels
from pykelihood.parameters import ConstantParameter, Parametrized
from pykelihood.stats_utils import ConditioningMethod

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


class Distribution(Parametrized):
    def __hash__(self):
        return (self.__class__.__name__,) + self.params

    @abstractmethod
    def rvs(self, size: int):
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

    def log_likelihood(
        self,
        data: Union[np.array, pd.Series],
        penalty: Callable = ConditioningMethod.no_conditioning,
        *args,
        **kwds,
    ):
        res = self.logpdf(data, *args, **kwds)
        return np.sum(res) - penalty(data, self)

    def opposite_log_likelihood(
        self,
        data: Union[np.array, pd.Series],
        penalty: Callable = ConditioningMethod.no_conditioning,
        *args,
        **kwds,
    ):
        return -self.log_likelihood(data, penalty, *args, **kwds)

    def _process_fit_params(self, **kwds):
        param_dict = self.param_dict.copy()
        for name in param_dict:
            for name_fixed_param, value_fixed_param in kwds.items():
                if name_fixed_param == name:
                    param_dict[name] = ConstantParameter(value_fixed_param)
                elif name_fixed_param.startswith(name):
                    subparam = name_fixed_param.replace(f"{name}_", "")
                    subdic = param_dict[name].param_dict.copy()
                    subdic[subparam] = ConstantParameter(value_fixed_param)
                    param_dict[name] = param_dict[name].with_params(subdic.values())
        return param_dict

    @classmethod
    def fit(
        cls,
        data,
        x0=None,
        penalty=ConditioningMethod.no_conditioning,
        **fixed_values,
    ):
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
            raise ValueError(
                f"Expected {len(init.optimisation_params)} values in x0, got {len(x0)}"
            )

        def to_minimize(x):
            o = init.with_params(x)
            return o.opposite_log_likelihood(data, penalty=penalty)

        res = minimize(to_minimize, x0, method="Nelder-Mead")
        return init.with_params(res.x)

    def fit_instance(
        self,
        data,
        penalty: Callable = ConditioningMethod.no_conditioning,
        fixed_params=None,
        **kwds,
    ):
        param_dict = self._process_fit_params(**(fixed_params or {}))
        kwds.update(param_dict)
        return self.fit(data, penalty=penalty, **kwds)


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
        if item not in ("pdf", "logpdf", "cdf", "logcdf", "ppf", "isf", "sf", "logsf"):
            return super(ScipyDistribution, self).__getattr__(item)
        f = getattr(self.base_module, item)
        g = partial(self._wrapper, f)
        g = self._correct_trends(g)
        g = cachetools.cached(
            self._cache, key=lambda x: hash((item, hash_with_series(x)))
        )(g)
        self.__dict__[item] = g
        return g


class Uniform(ScipyDistribution):
    params_names = ("loc", "scale")
    base_module = uniform

    def __init__(self, loc=0.0, scale=1.0):
        super(Uniform, self).__init__(loc, scale)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None):
        return f(x, ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None):
        return uniform.rvs(ifnone(loc, self.loc()), ifnone(scale, self.scale()), size)


class Exponential(ScipyDistribution):
    params_names = ("loc", "rate")
    base_module = expon

    def __init__(self, loc=0.0, rate=1.0):
        super(Exponential, self).__init__(loc, rate)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, rate=None):
        if rate is not None:
            rate = 1 / rate
        return f(x, ifnone(loc, self.loc()), ifnone(rate, 1 / self.rate()))

    def rvs(self, size, loc=None, rate=None):
        return self.base_module.rvs(
            ifnone(loc, self.loc()), ifnone(rate, 1 / self.rate()), size
        )


class Gamma(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = gamma

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super(Gamma, self).__init__(loc, scale, shape)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, shape=None):
        return f(
            x,
            ifnone(shape, self.shape()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
        )

    def rvs(self, size, loc=None, scale=None, shape=None):
        return self.base_module.rvs(
            ifnone(shape, self.shape()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
            size,
        )


class Pareto(ScipyDistribution):
    params_names = ("loc", "scale", "alpha")
    base_module = pareto

    def __init__(self, loc=0.0, scale=1.0, alpha=1.0):
        super(Pareto, self).__init__(loc, scale, alpha)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, alpha=None):
        return f(
            x,
            ifnone(alpha, self.alpha()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
        )

    def rvs(self, size, loc=None, scale=None, alpha=None):
        return self.base_module.rvs(
            ifnone(alpha, self.alpha()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
            size,
        )


class Beta(ScipyDistribution):
    params_names = ("loc", "scale", "alpha", "beta")
    base_module = beta

    def __init__(self, loc=0.0, scale=1.0, alpha=2.0, beta=1.0):
        super(Beta, self).__init__(loc, scale, alpha, beta)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, alpha=None, beta=None):
        return f(
            x,
            ifnone(alpha, self.alpha()),
            ifnone(beta, self.beta()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
        )

    def rvs(self, size, loc=None, scale=None, alpha=None, beta=None):
        return self.base_module.rvs(
            ifnone(alpha, self.alpha()),
            ifnone(beta, self.beta()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
            size,
        )


class Normal(ScipyDistribution):
    params_names = ("loc", "scale")
    base_module = norm

    def __init__(self, loc=0.0, scale=1.0):
        super(Normal, self).__init__(loc, scale)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None):
        return f(x, ifnone(loc, self.loc()), ifnone(scale, self.scale()))

    def rvs(self, size, loc=None, scale=None):
        return self.base_module.rvs(
            ifnone(loc, self.loc()), ifnone(scale, self.scale()), size
        )


class GEV(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = genextreme

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
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
        return f(
            x,
            ifnone(shape, -self.shape()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
        )

    def rvs(self, size, shape=None, loc=None, scale=None):
        if shape is not None:
            shape = -shape
        return self.base_module.rvs(
            ifnone(shape, -self.shape()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
            size,
        )


class MixtureExponentialModel(Distribution):
    params_names = ("theta",)

    def __init__(self, theta=0.99):
        super(MixtureExponentialModel, self).__init__(theta)
        self._cache = {}

    def rvs(self, size):
        theta = self.theta()
        uniforms = Uniform().rvs(size)
        realisations = []
        for uniform in uniforms:
            if uniform <= 1 - theta:
                realisations.append(0.0)
            else:
                realisations.append(Exponential(rate=theta).rvs(1))
        return realisations

    def pdf(self, x):
        theta = self.theta()
        return np.where(x > 0.0, theta * Exponential(rate=theta).pdf(x), 1 - theta)

    def cdf(self, x):
        theta = self.theta()
        return np.where(x > 0.0, theta * Exponential(rate=theta).cdf(x), 1 - theta)

    def isf(self, q):
        theta = self.theta()
        return np.where(q != 1 - theta, Exponential(rate=theta).isf(q / theta), 0.0)


class GPD(ScipyDistribution):
    params_names = ("loc", "scale", "shape")
    base_module = genpareto

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super(GPD, self).__init__(loc, scale, shape)
        self._cache = {}

    def _wrapper(self, f, x, loc=None, scale=None, shape=None):
        return f(
            x,
            ifnone(shape, self.shape()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
        )

    def rvs(self, size, loc=None, scale=None, shape=None):
        return self.base_module.rvs(
            ifnone(shape, self.shape()),
            ifnone(loc, self.loc()),
            ifnone(scale, self.scale()),
            size,
        )


class ExtendedGPD(Distribution):
    params_names = ("loc", "scale", "shape")

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super(ExtendedGPD, self).__init__(loc, scale, shape)
        if shape == 0.0:
            raise ValueError(
                "For this distribution, the shape parameter must not be zero."
            )

    def rvs(self, size: int):
        return self.inverse_cdf(Uniform().rvs(size))

    def cdf(self, x: Union[np.array, float]):
        return 1 - np.clip(
            1 + self.shape * ((x - self.loc) / self.scale), 0, a_max=None
        ) ** (-1 / self.shape)

    def inverse_cdf(self, q: float):
        return self.loc + self.scale * ((1 - q) ** (-self.shape) - 1) / self.shape

    def pdf(self, x: Union[np.array, float]):
        return (1 / self.scale) * np.clip(
            1 + self.shape * ((x - self.loc) / self.scale), 0, a_max=None
        ) ** (-1 / self.shape - 1)

    def isf(self, q: float):
        return self.loc + self.scale * (q ** (-self.shape) - 1) / self.shape


class PointProcess(Distribution):
    def __init__(
        self,
        threshold: int,
        intensity,
        jumps_size_distribution: Distribution = GPD(),
        remaining_points_distribution: Distribution = Uniform(
            ConstantParameter(0.0), ConstantParameter(1.0)
        ),
        npp=365.25,
    ):
        """

        :param threshold: defines what is a jump in term of exceeding of a certain threshold
        :param jumps_size_distribution: the jump size distribution
        :param intensity: the inter-arrival intensity lambda which can depend on time
        (non homogeneous Poisson Process) or history (Hawkes Process).
         If not provided, Time is automatically set to indices and renormalized between 0 and 1.
        :param remaining_points_distribution: optional, handles the points that are not jumps (appearing with frequency
        1-lambda on average)
        :param npp: number of observations per period, used to compute return level estimates over k years
        """
        self.iat = Exponential(loc=ConstantParameter(0.0), rate=intensity)
        super(PointProcess, self).__init__(
            *(
                self.iat.params
                + jumps_size_distribution.params
                + remaining_points_distribution.params
            )
        )
        self.jumps_size = jumps_size_distribution
        self.remaining_points = remaining_points_distribution
        self.threshold = threshold
        self.intensity = intensity
        self.frequency_exceedances = 1 / intensity()
        self.npp = npp

    @property
    def params_names(self):
        IAT_names = [f"iat_{p}" for p in self.iat.params_names]
        JS_names = [f"js_{p}" for p in self.jumps_size.params_names]
        RemainingPoints_names = [f"rp_{p}" for p in self.remaining_points.params_names]
        return tuple(IAT_names) + tuple(JS_names) + tuple(RemainingPoints_names)

    def with_params(self, new_params):
        new_params = iter(new_params)
        iat = self.iat.with_params(new_params)
        js = self.jumps_size.with_params(new_params)
        rp = self.remaining_points.with_params(new_params)
        return type(self)(self.threshold, iat.rate, js, rp, self.npp)

    def _process_fit_params(self, **kwds):
        IAT_kwds = {
            k.replace("iat_", ""): v for (k, v) in kwds.items() if k.startswith("iat_")
        }
        JS_kwds = {
            k.replace("js_", ""): v for (k, v) in kwds.items() if k.startswith("js_")
        }
        RP_kwds = {
            k.replace("rp_", ""): v for (k, v) in kwds.items() if k.startswith("rp_")
        }
        IAT_params = self.iat._process_fit_params(**IAT_kwds)
        JS_params = self.jumps_size._process_fit_params(**JS_kwds)
        RP_params = self.remaining_points._process_fit_params(**RP_kwds)
        if len(kwds) > len(IAT_kwds) + len(JS_kwds) + len(RP_kwds):
            raise ValueError(f"Unexpected parameters encountered in keywords: {kwds}")
        return IAT_params, JS_params, RP_params

    def fit_instance(
        self,
        data,
        penalty: Callable = ConditioningMethod.no_conditioning,
        fixed_params=None,
        **kwds,
    ):
        kwds.update(**(fixed_params or {}))
        return self.fit(data, penalty=penalty, **kwds)

    def fit(
        self,
        data,
        penalty=ConditioningMethod.no_conditioning,
        x0=None,
        opt_method="Nelder-Mead",
        plot=False,
        *args,
        **kwds,
    ):
        params_iat, params_js, params_rp = self._process_fit_params(**kwds)
        obj = type(self)(
            self.threshold,
            type(self.iat)(**params_iat).rate,
            type(self.jumps_size)(**params_js),
            type(self.remaining_points)(**params_rp),
        )
        # Three different fittings to avoid numerical instability
        inter_arrival_times = np.diff(
            np.asarray(data >= self.threshold).nonzero()[0] / len(data)
        )
        inter_arrival_times = np.insert(inter_arrival_times, 0, 0)
        distribution_iat = obj.iat.fit(inter_arrival_times, **obj.iat.param_dict)
        pot = data[data >= self.threshold]
        distribution_jump_size = obj.jumps_size.fit(pot, **obj.jumps_size.param_dict)
        remaining_points = data[(~data.isin(pot)) & data > 0]
        distribution_remaining_points = obj.remaining_points.fit(
            remaining_points, **obj.remaining_points.param_dict
        )
        res = list(
            tuple(distribution_iat.optimisation_params)
            + tuple(distribution_jump_size.optimisation_params)
            + tuple(distribution_remaining_points.optimisation_params)
        )
        fitted_distribution = obj.with_params(res)
        fitted_distribution.frequency_exceedances = (
            fitted_distribution.intensity() / len(data)
        )
        return fitted_distribution

    def show_diagnostic_plot(
        self, data, time_scale=("d", pd.to_datetime("1961-01-01"))
    ):
        """

        :param data: empirical observations used for the fit
        :param time_scale: tuple (a,b) with a being the frequency of observations per period (ie daily data over a year:
        a="d") and b the first period (for data starting in year 1961, b=pd.to_datetime("1961-01-01"))
        :return: plots
        """
        import matplotlib.pyplot as plt

        n_simul = 500
        simul, exceedances = self.rvs(
            n_simul, fixed_number_of_exceedances=len(data[data >= self.threshold])
        )
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])

        # Cumulative Number of Exceedances
        simu = pd.concat([pd.Series(d.index) for d in exceedances], axis=1)
        pot = data[data >= self.threshold]
        realized = (pot >= self.threshold).cumsum()
        simu.index = simu.index + 1
        mean = simu.mean(axis=1)
        std = simu.std(axis=1)

        def swap_axes(s):
            return pd.Series(s.index.values, index=s)

        mean_ = swap_axes(mean)
        x = pd.to_datetime(
            len(data) * realized.index, unit=time_scale[0], origin=time_scale[1]
        )
        x1 = pd.to_datetime(
            len(data) * mean_.index, unit=time_scale[0], origin=time_scale[1]
        )
        x2 = pd.to_datetime(
            len(data) * (mean + 2 * std), unit=time_scale[0], origin=time_scale[1]
        )
        x3 = pd.to_datetime(
            len(data) * (mean - 2 * std), unit=time_scale[0], origin=time_scale[1]
        )
        realized.index = x
        mean_.index = x1
        ax1.plot(realized, label="Realized", color="black")
        ax1.plot(mean_, label=f"Simulated ({n_simul} points)", color="r")
        ax1.fill_betweenx(y=mean_, x1=x2, x2=x3, color="r", alpha=0.2)
        ax1.set_title("Cumulative Exceedances over Threshold")
        ax1.set_ylabel("Number of Exceedances")
        ax1.set_xlabel("Time")
        ax1.legend()

        # Peaks Over Threshold
        realized_cdf = pot.value_counts(normalize=True).sort_index().cumsum()
        if len(self.jumps_size.params) == len(self.jumps_size.flattened_params):
            theoretical = self.jumps_size.cdf(realized_cdf.index)
            ax2.plot(realized_cdf, theoretical)
            ax2.plot(realized_cdf, realized_cdf)
            ax2.set_title("Quantile Plot Peaks Over Threshold")
            ax2.set_xlabel("Empirical")
            ax2.set_ylabel("Theoretical")
        else:
            simulated = pd.concat(
                [
                    e.transform(lambda x: round(x, 1), axis=0)
                    .value_counts(normalize=True)
                    .sort_index()
                    .cumsum()
                    for e in exceedances
                ],
                axis=1,
            )
            mean = (
                pd.concat([simulated.mean(axis=1), realized_cdf], axis=1)
                .sort_index()
                .ffill()
                .loc[realized_cdf.index][0]
            )
            ax2.plot(realized_cdf, mean)
            ax2.plot(realized_cdf, realized_cdf)
            ax2.set_title("Quantile Plot Peaks Over Threshold")
            ax2.set_xlabel("Empirical")
            ax2.set_ylabel(f"Simulated ({n_simul} points)")

        # Points Under Threshold
        ut = data[data < self.threshold]
        realized_cdf = ut.value_counts(normalize=True).sort_index().cumsum()
        theoretical = self.remaining_points.cdf(realized_cdf.index)
        ax3.plot(realized_cdf, theoretical)
        ax3.plot(realized_cdf, realized_cdf)
        ax3.set_title("Quantile Plot Points Under Threshold")
        ax3.set_xlabel("Empirical")
        ax3.set_ylabel("Theoretical")

        # Return-level graph
        nb_period = int(round(len(data) / self.npp, 0))
        simulated_points = pd.concat(simul, axis=1)
        simulated_points.index = nb_period * simulated_points.index
        data_rescaled = data.copy()
        data_rescaled.index = nb_period * data_rescaled.index
        if not isinstance(self.intensity(), float):
            freq_series = pd.Series(
                self.frequency_exceedances,
                index=self.intensity.f.args[0],
                name="intensity",
            )
            freq_series.index = nb_period * freq_series.index
        else:
            freq_series = pd.Series(
                [self.frequency_exceedances] * len(pot),
                index=nb_period * pot.index,
                name="intensity",
            )
        true_vs_realized = []
        rle = self.isf(1.0)
        for i, return_level_estimate in zip(freq_series.index, rle):
            true_return_level = data_rescaled.loc[i - 1 : i].max()
            simulated_mean = simulated_points.loc[i - 1 : i].max().mean()
            simulated_std = simulated_points.loc[i - 1 : i].std().mean()
            true_vs_realized.append(
                pd.DataFrame(
                    [
                        [
                            true_return_level,
                            return_level_estimate,
                            simulated_mean,
                            simulated_mean - 2 * simulated_std,
                            simulated_mean + 2 * simulated_std,
                        ]
                    ],
                    columns=[
                        "Realized",
                        "Estimated",
                        f"Simulated ({n_simul} Points)",
                        "SimulBInf",
                        "SimulBSup",
                    ],
                    index=[
                        time_scale[1] + pd.to_timedelta(f"{self.npp*i} {time_scale[0]}")
                    ],
                )
            )
        true_vs_realized = pd.concat(true_vs_realized)

        def moving_average(x):
            ret = np.cumsum(x)
            bool = (x > 0).cumsum()
            return ret / bool

        for col, color in zip(true_vs_realized.columns, ["black", "g", "r", "", ""]):
            true_vs_realized[col] = moving_average(true_vs_realized[col])
            if col not in ["SimulBInf", "SimulBSup"]:
                ax4.plot(true_vs_realized[col], label=col, color=color)
        inf_bound, sup_bound = (
            true_vs_realized["SimulBInf"],
            true_vs_realized["SimulBSup"],
        )
        ax4.fill_between(
            x=true_vs_realized.index, y1=inf_bound, y2=sup_bound, color="r", alpha=0.2
        )
        ax4.set_title("1 year Return level Moving Average")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Return level")
        ax4.legend()
        return fig

    def rvs(self, size, fixed_number_of_exceedances=None, show_diagnostic_plots=True):
        from pykelihood.samplers import HawkesByThinningModified, PoissonByThinning

        simul = []
        exceedances_raw = []
        for _ in range(size):
            if fixed_number_of_exceedances:
                nb_exceedances = fixed_number_of_exceedances
                expo = np.cumsum(self.iat.rvs(size=nb_exceedances))
            else:
                if (
                    self.intensity.original_f.func.__name__
                    == kernels.hawkes_with_exp_kernel.__name__
                ):
                    mu, alpha, theta = self.intensity.params
                    expo = HawkesByThinningModified(1.0, mu, alpha, theta)
                else:
                    expo = PoissonByThinning(
                        1.0, self.intensity, np.max(self.intensity())
                    )
                nb_exceedances = len(expo)
            exceedances = np.array(self.jumps_size.rvs(nb_exceedances))
            exceedances_raw.append(pd.Series(exceedances, index=expo, name=f"{_}"))
            total_number_of_points = int(
                np.mean(self.intensity()) / np.mean(self.frequency_exceedances)
            )
            non_exceedances = total_number_of_points - nb_exceedances
            under_threshold_points = self.remaining_points.rvs(size=non_exceedances)
            indices_utp = np.random.choice(
                list(set(np.linspace(0, 1, total_number_of_points)).difference(expo)),
                non_exceedances,
                replace=False,
            )
            points_over_threshold = pd.Series(exceedances, index=expo, name=f"{_}")
            points_under_threshold = pd.Series(
                under_threshold_points, index=indices_utp, name=f"{_}"
            )
            simul.append(
                pd.concat([points_under_threshold, points_over_threshold]).sort_index()
            )
        return simul, exceedances_raw

    def cdf(self, x):
        freq = self.frequency_exceedances
        if not isinstance(x, pd.Series) and not isinstance(freq, float):
            raise TypeError(
                "The information of the covariate value is missing, please input a pandas Series"
                "with index corresponding to this information."
            )
        if isinstance(x, pd.Series) and not isinstance(freq, float):
            freq_series = pd.Series(
                freq / len(x), index=self.intensity.f.args[0], name="intensity"
            )
            freq = (
                pd.concat([freq_series, x], axis=1)
                .sort_index()
                .bfill()
                .ffill()["intensity"]
            )
        res = np.where(
            x < self.threshold,
            self.remaining_points.cdf(x) * (1 - freq),
            freq * self.jumps_size.cdf(x),
        )
        return res

    def isf(self, q):
        freq = self.frequency_exceedances
        return_level_proba = q / self.npp
        res = np.where(
            return_level_proba > freq,
            self.remaining_points.isf(return_level_proba / (1 - freq)),
            self.jumps_size.isf(return_level_proba / freq),
        )
        return res

    def pdf(self, x):
        freq = self.frequency_exceedances
        if not isinstance(x, pd.Series) and not isinstance(freq, float):
            raise TypeError(
                "The information of the covariate value is missing, please input a pandas Series"
                "with index corresponding to this information."
            )
        if isinstance(x, pd.Series) and not isinstance(freq, float):
            freq_series = pd.Series(
                freq, index=self.intensity.f.args[0], name="intensity"
            )
            freq = (
                pd.concat([freq_series, x], axis=1)
                .sort_index()
                .bfill()
                .ffill()["intensity"]
            )
        res = np.where(
            x < self.threshold,
            (1 - freq) * self.remaining_points.pdf(x),
            self.jumps_size.pdf(x) * freq,
        )
        return res


class CompositionDistribution(Distribution):
    def __init__(self, d1: Distribution, d2: Distribution):
        """
        Distribution of cdf GoF
        :param d1: distribution aimed at replacing the uniform sampling in the formula used
        to generate variables with distribution d2 (G in F^{-1}(G^{-1}(U))
        :param d2: main objective distribution, especially aimed at determining the asymptotic behaviour.
        """
        super(CompositionDistribution, self).__init__(*(d1.params + d2.params))
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
        d1_kwds = {
            k.replace("d1_", ""): v for (k, v) in kwds.items() if k.startswith("d1_")
        }
        d2_kwds = {
            k.replace("d2_", ""): v for (k, v) in kwds.items() if k.startswith("d2_")
        }
        d1_params = self.d1._process_fit_params(**d1_kwds)
        d2_params = self.d2._process_fit_params(**d2_kwds)
        if len(kwds) > len(d1_kwds) + len(d2_kwds):
            raise ValueError(f"Unexpected parameters encountered in keywords: {kwds}")
        return d1_params, d2_params

    def fit_instance(
        self,
        data,
        penalty: Callable = ConditioningMethod.no_conditioning,
        fixed_params=None,
        **kwds,
    ):
        kwds.update(**(fixed_params or {}))
        return self.fit(data, penalty=penalty, **kwds)

    def fit(
        self,
        data: pd.Series,
        penalty=ConditioningMethod.no_conditioning,
        x0=None,
        opt_method="Nelder-Mead",
        *args,
        **kwds,
    ):
        params_d1, params_d2 = self._process_fit_params(**kwds)
        obj = type(self)(type(self.d1)(**params_d1), type(self.d2)(**params_d2))
        x0 = obj.optimisation_params if x0 is None else x0

        def to_minimize(var_params):
            return obj.with_params(var_params).opposite_log_likelihood(data, penalty)

        result = minimize(
            to_minimize,
            method=opt_method,
            options={"adaptive": True, "maxfev": 1000},
            x0=x0,
        )
        res = list(result.x)
        return obj.with_params(res)
