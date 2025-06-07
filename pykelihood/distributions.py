from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
import scipy.special
from packaging.version import Version
from scipy import stats
from scipy.optimize import OptimizeResult, minimize

from pykelihood.generic_types import Obs
from pykelihood.metrics import opposite_log_likelihood
from pykelihood.parameters import ConstantParameter, Parametrized, ensure_parametrized
from pykelihood.utils import ifnone

if TYPE_CHECKING:
    from typing import Self


_T = TypeVar("_T")
SomeDistribution = TypeVar("SomeDistribution", bound="Distribution")

EULER = -scipy.special.psi(1)


class Distribution(Parametrized):
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
    fit_instance(data, score=opposite_log_likelihood, x0: Sequence[float] = None, scipy_args: Optional[Dict] = None, **fixed_values)
        Fit the instance to the data.
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

    @classmethod
    def fit(
        cls: type[SomeDistribution],
        data: Obs,
        x0: Sequence[float] | None = None,
        score: Callable[[Distribution, Obs], float] = opposite_log_likelihood,
        scipy_args: dict | None = None,
        **fixed_values,
    ) -> Fit[SomeDistribution]:
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
        The result of the fit
        """
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
        data = init._apply_constraints(data)

        if x0 is None:
            x0 = [x.value for x in init.optimisation_params]
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
        data: Obs,
        score=opposite_log_likelihood,
        x0: Sequence[float] | None = None,
        scipy_args: dict | None = None,
        **fixed_values,
    ) -> Fit[Self]:
        """
        Fit the instance to the data.

        Parameters
        ----------
        data : Obs
            Data to fit the instance to.
        score : Callable[["Distribution", Obs], float], optional
            Scoring function, by default opposite_log_likelihood.
        x0 : Sequence[float], optional
            Initial guess for the parameters, by default None.
        scipy_args : Optional[Dict], optional
            Additional arguments for scipy.optimize.minimize, by default None.
        fixed_values : dict
            Fixed values for the parameters.

        Returns
        -------
        Distribution
            Fitted instance.
        """
        param_dict = self._process_fit_params(**fixed_values)
        return self.fit(data, score=score, x0=x0, scipy_args=scipy_args, **param_dict)


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


class AvoidAbstractMixin:
    """
    Mixin to avoid abstract methods.

    Methods
    -------
    __getattribute__(item)
        Get the attribute, avoiding abstract methods.
    """

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
    """
    Base class for distributions using scipy.

    Methods
    -------
    rvs(size=None, random_state=None, **kwargs)
        Generate random variates.
    _wrapper(f, x, **extra_args)
        Wrapper for scipy functions.
    __getattr__(item)
        Get the attribute, wrapping scipy functions.
    """

    base_module: stats.rv_continuous

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
            return super().__getattr__(item)
        f = getattr(self.base_module, item)
        g = partial(self._wrapper, f)
        self.__dict__[item] = g
        return g


class Exponential(ScipyDistribution):
    """
    Exponential distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    rate : float, optional
        Rate parameter, by default 1.0.
    """

    params_names = ("loc", "rate")
    base_module = stats.expon

    def __init__(self, loc=0.0, rate=1.0):
        super().__init__(loc, rate)

    def _to_scipy_args(self, loc=None, rate=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        rate : float, optional
            Rate parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        if rate is not None:
            rate = 1 / rate
        return {"loc": ifnone(loc, self.loc()), "scale": ifnone(rate, 1 / self.rate())}


class Gamma(ScipyDistribution):
    """
    Gamma distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    shape : float, optional
        Shape parameter, by default 0.0.
    """

    params_names = ("loc", "scale", "shape")
    base_module = stats.gamma

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super().__init__(loc, scale, shape)

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        shape : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "a": ifnone(shape, self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class Pareto(ScipyDistribution):
    """
    Pareto distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    alpha : float, optional
        Shape parameter, by default 1.0.
    """

    params_names = ("loc", "scale", "alpha")
    base_module = stats.pareto

    def __init__(self, loc=0.0, scale=1.0, alpha=1.0):
        super().__init__(loc, scale, alpha)

    def _to_scipy_args(self, loc=None, scale=None, alpha=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        alpha : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "c": ifnone(alpha, self.alpha()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class Beta(ScipyDistribution):
    """
    Beta distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    alpha : float, optional
        Alpha parameter, by default 2.0.
    beta : float, optional
        Beta parameter, by default 1.0.
    """

    params_names = ("loc", "scale", "alpha", "beta")
    base_module = stats.beta

    def __init__(self, loc=0.0, scale=1.0, alpha=2.0, beta=1.0):
        super().__init__(loc, scale, alpha, beta)

    def _to_scipy_args(self, loc=None, scale=None, alpha=None, beta=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        alpha : float, optional
            Alpha parameter, by default None.
        beta : float, optional
            Beta parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "a": ifnone(alpha, self.alpha()),
            "b": ifnone(beta, self.beta()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class GEV(ScipyDistribution):
    """
    Generalized Extreme Value (GEV) distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    shape : float, optional
        Shape parameter, by default 0.0.

    Notes
    -----
    This version of the Generalized Extreme Value distribution (GEV) does not
    have parameters `c`, `loc`, `scale` but `loc`, `scale` and `shape` where shape
    is `-c`.
    """

    params_names = ("loc", "scale", "shape")
    base_module = stats.genextreme

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super().__init__(loc, scale, shape)

    def lb_shape(self, data):
        """
        Calculate the lower bound of the shape parameter.

        Parameters
        ----------
        data : array-like
            Data to calculate the lower bound.

        Returns
        -------
        float
            Lower bound of the shape parameter.
        """
        x_min = data.min()
        x_max = data.max()
        if x_min * x_max < 0:
            return -np.inf
        elif x_min > 0:
            return self.scale / (x_max - self.loc())
        else:
            return self.scale / (x_min - self.loc())

    def ub_shape(self, data):
        """
        Calculate the upper bound of the shape parameter.

        Parameters
        ----------
        data : array-like
            Data to calculate the upper bound.

        Returns
        -------
        float
            Upper bound of the shape parameter.
        """
        x_min = data.min()
        x_max = data.max()
        if x_min * x_max < 0:
            return np.inf
        elif x_min > 0:
            return self.scale / (x_min - self.loc())
        else:
            return self.scale / (x_max - self.loc())

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        shape : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        if shape is not None:
            shape = -shape
        return {
            "c": ifnone(shape, -self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


class GPD(ScipyDistribution):
    """
    Generalized Pareto Distribution (GPD).

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.
    shape : float, optional
        Shape parameter, by default 0.0.
    """

    params_names = ("loc", "scale", "shape")
    base_module = stats.genpareto

    def __init__(self, loc=0.0, scale=1.0, shape=0.0):
        super().__init__(loc, scale, shape)

    def _to_scipy_args(self, loc=None, scale=None, shape=None):
        """
        Convert to scipy arguments.

        Parameters
        ----------
        loc : float, optional
            Location parameter, by default None.
        scale : float, optional
            Scale parameter, by default None.
        shape : float, optional
            Shape parameter, by default None.

        Returns
        -------
        dict
            Dictionary of scipy arguments.
        """
        return {
            "c": ifnone(shape, self.shape()),
            "loc": ifnone(loc, self.loc()),
            "scale": ifnone(scale, self.scale()),
        }


def _name_from_scipy_dist(scipy_dist: stats.rv_continuous) -> str:
    """Generate a name for the distribution based on the scipy distribution class."""
    scipy_dist_name = type(scipy_dist).__name__.removesuffix("_gen")
    return "".join(map(str.capitalize, scipy_dist_name.split("_")))


def _wrap_scipy_distribution(
    scipy_dist: stats.rv_continuous,
) -> type[ScipyDistribution]:
    """Wrap a scipy distribution class to create a ScipyDistribution subclass."""
    scipy_dist_name = type(scipy_dist).__name__.removesuffix("_gen")
    clean_dist_name = _name_from_scipy_dist(scipy_dist)
    params_names = ("loc", "scale") + tuple(
        scipy_dist.shapes.split(", ") if scipy_dist.shapes else ()
    )

    docstring = f"""\
    {clean_dist_name} distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.\
    """

    def format_param_docstring(param: str) -> str:
        return f"""
    {param} : float, mandatory
        Shape parameter. See the SciPy documentation for the {scipy_dist_name} distribution for details.\
        """

    for param in params_names[2:]:
        docstring += format_param_docstring(param)

    def __init__(self, loc=0.0, scale=1.0, **kwargs):
        shape_args = params_names[2:]
        for arg in shape_args:
            if arg not in kwargs:
                raise ValueError(f"Missing shape parameter: {arg}")
        args = [kwargs[a] for a in shape_args]
        ScipyDistribution.__init__(self, loc, scale, *args)

    def _to_scipy_args(self, **kwargs):
        return {k: kwargs.get(k, getattr(self, k)()) for k in self.params_names}

    return type(
        clean_dist_name,
        (ScipyDistribution,),
        {
            "base_module": scipy_dist,
            "params_names": params_names,
            "__init__": __init__,
            "_to_scipy_args": _to_scipy_args,
            "__doc__": docstring,
        },
    )


Alpha = _wrap_scipy_distribution(stats.alpha)
Anglit = _wrap_scipy_distribution(stats.anglit)
Arcsine = _wrap_scipy_distribution(stats.arcsine)
Argus = _wrap_scipy_distribution(stats.argus)
# Beta = _wrap_scipy_distribution(stats.beta)
Betaprime = _wrap_scipy_distribution(stats.betaprime)
Bradford = _wrap_scipy_distribution(stats.bradford)
Burr = _wrap_scipy_distribution(stats.burr)
Burr12 = _wrap_scipy_distribution(stats.burr12)
Cauchy = _wrap_scipy_distribution(stats.cauchy)
Chi = _wrap_scipy_distribution(stats.chi)
Chi2 = _wrap_scipy_distribution(stats.chi2)
Cosine = _wrap_scipy_distribution(stats.cosine)
Crystalball = _wrap_scipy_distribution(stats.crystalball)
Dgamma = _wrap_scipy_distribution(stats.dgamma)
Dweibull = _wrap_scipy_distribution(stats.dweibull)
Erlang = _wrap_scipy_distribution(stats.erlang)
Expon = _wrap_scipy_distribution(stats.expon)
Exponnorm = _wrap_scipy_distribution(stats.exponnorm)
Exponpow = _wrap_scipy_distribution(stats.exponpow)
Exponweib = _wrap_scipy_distribution(stats.exponweib)
F = _wrap_scipy_distribution(stats.f)
Fatiguelife = _wrap_scipy_distribution(stats.fatiguelife)
Fisk = _wrap_scipy_distribution(stats.fisk)
Foldcauchy = _wrap_scipy_distribution(stats.foldcauchy)
Foldnorm = _wrap_scipy_distribution(stats.foldnorm)
# Gamma = _wrap_scipy_distribution(stats.gamma)
Gausshyper = _wrap_scipy_distribution(stats.gausshyper)
Genexpon = _wrap_scipy_distribution(stats.genexpon)
Genextreme = _wrap_scipy_distribution(stats.genextreme)
Gengamma = _wrap_scipy_distribution(stats.gengamma)
Genhalflogistic = _wrap_scipy_distribution(stats.genhalflogistic)
Genhyperbolic = _wrap_scipy_distribution(stats.genhyperbolic)
Geninvgauss = _wrap_scipy_distribution(stats.geninvgauss)
Genlogistic = _wrap_scipy_distribution(stats.genlogistic)
Gennorm = _wrap_scipy_distribution(stats.gennorm)
Genpareto = _wrap_scipy_distribution(stats.genpareto)
Gibrat = _wrap_scipy_distribution(stats.gibrat)
Gompertz = _wrap_scipy_distribution(stats.gompertz)
GumbelL = _wrap_scipy_distribution(stats.gumbel_l)
GumbelR = _wrap_scipy_distribution(stats.gumbel_r)
Halfcauchy = _wrap_scipy_distribution(stats.halfcauchy)
Halfgennorm = _wrap_scipy_distribution(stats.halfgennorm)
Halflogistic = _wrap_scipy_distribution(stats.halflogistic)
Halfnorm = _wrap_scipy_distribution(stats.halfnorm)
Hypsecant = _wrap_scipy_distribution(stats.hypsecant)
Invgamma = _wrap_scipy_distribution(stats.invgamma)
Invgauss = _wrap_scipy_distribution(stats.invgauss)
Invweibull = _wrap_scipy_distribution(stats.invweibull)
JfSkewT = _wrap_scipy_distribution(stats.jf_skew_t)
Johnsonsb = _wrap_scipy_distribution(stats.johnsonsb)
Johnsonsu = _wrap_scipy_distribution(stats.johnsonsu)
Kappa3 = _wrap_scipy_distribution(stats.kappa3)
Kappa4 = _wrap_scipy_distribution(stats.kappa4)
Ksone = _wrap_scipy_distribution(stats.ksone)
Kstwo = _wrap_scipy_distribution(stats.kstwo)
Kstwobign = _wrap_scipy_distribution(stats.kstwobign)
Laplace = _wrap_scipy_distribution(stats.laplace)
LaplaceAsymmetric = _wrap_scipy_distribution(stats.laplace_asymmetric)
Levy = _wrap_scipy_distribution(stats.levy)
LevyL = _wrap_scipy_distribution(stats.levy_l)
LevyStable = _wrap_scipy_distribution(stats.levy_stable)
Loggamma = _wrap_scipy_distribution(stats.loggamma)
Logistic = _wrap_scipy_distribution(stats.logistic)
Loglaplace = _wrap_scipy_distribution(stats.loglaplace)
Lognorm = _wrap_scipy_distribution(stats.lognorm)
Lomax = _wrap_scipy_distribution(stats.lomax)
Maxwell = _wrap_scipy_distribution(stats.maxwell)
Mielke = _wrap_scipy_distribution(stats.mielke)
Moyal = _wrap_scipy_distribution(stats.moyal)
Nakagami = _wrap_scipy_distribution(stats.nakagami)
Ncf = _wrap_scipy_distribution(stats.ncf)
Nct = _wrap_scipy_distribution(stats.nct)
Ncx2 = _wrap_scipy_distribution(stats.ncx2)
Norm = _wrap_scipy_distribution(stats.norm)
Normal = Norm  # alias for backward compatibility
Norminvgauss = _wrap_scipy_distribution(stats.norminvgauss)
# Pareto = _wrap_scipy_distribution(stats.pareto)
Pearson3 = _wrap_scipy_distribution(stats.pearson3)
Powerlaw = _wrap_scipy_distribution(stats.powerlaw)
Powerlognorm = _wrap_scipy_distribution(stats.powerlognorm)
Powernorm = _wrap_scipy_distribution(stats.powernorm)
Rayleigh = _wrap_scipy_distribution(stats.rayleigh)
Rdist = _wrap_scipy_distribution(stats.rdist)
Recipinvgauss = _wrap_scipy_distribution(stats.recipinvgauss)
Loguniform = _wrap_scipy_distribution(stats.loguniform)
Reciprocal = _wrap_scipy_distribution(stats.reciprocal)
RelBreitwigner = _wrap_scipy_distribution(stats.rel_breitwigner)
Rice = _wrap_scipy_distribution(stats.rice)
Semicircular = _wrap_scipy_distribution(stats.semicircular)
Skewcauchy = _wrap_scipy_distribution(stats.skewcauchy)
Skewnorm = _wrap_scipy_distribution(stats.skewnorm)
StudentizedRange = _wrap_scipy_distribution(stats.studentized_range)
T = _wrap_scipy_distribution(stats.t)
Trapezoid = _wrap_scipy_distribution(stats.trapezoid)
Trapz = Trapezoid
Triang = _wrap_scipy_distribution(stats.triang)
Truncexpon = _wrap_scipy_distribution(stats.truncexpon)
Truncnorm = _wrap_scipy_distribution(stats.truncnorm)
Truncpareto = _wrap_scipy_distribution(stats.truncpareto)
TruncweibullMin = _wrap_scipy_distribution(stats.truncweibull_min)
Tukeylambda = _wrap_scipy_distribution(stats.tukeylambda)
Uniform = _wrap_scipy_distribution(stats.uniform)
Vonmises = _wrap_scipy_distribution(stats.vonmises)
VonmisesLine = _wrap_scipy_distribution(stats.vonmises_line)
Wald = _wrap_scipy_distribution(stats.wald)
WeibullMax = _wrap_scipy_distribution(stats.weibull_max)
WeibullMin = _wrap_scipy_distribution(stats.weibull_min)
Wrapcauchy = _wrap_scipy_distribution(stats.wrapcauchy)

if Version(scipy.__version__) >= Version("1.15.0"):
    DparetoLognorm = _wrap_scipy_distribution(stats.dpareto_lognorm)
    Landau = _wrap_scipy_distribution(stats.landau)
    Irwinhall = _wrap_scipy_distribution(stats.irwinhall)


class TruncatedDistribution(Distribution):
    """
    Truncated distribution.

    Parameters
    ----------
    distribution : Distribution
        The base distribution to truncate.
    lower_bound : float, optional
        Lower bound of the distribution, by default -np.inf.
    upper_bound : float, optional
        Upper bound of the distribution, by default np.inf.

    Raises
    ------
    ValueError
        If the lower and upper bounds are equal.
    """

    params_names = ("distribution",)

    def __init__(
        self, distribution: Distribution, lower_bound=-np.inf, upper_bound=np.inf
    ):
        if upper_bound == lower_bound:
            raise ValueError("Both bounds are equal.")
        super().__init__(distribution)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        lower_cdf = self.distribution.cdf(self.lower_bound)
        upper_cdf = self.distribution.cdf(self.upper_bound)
        self._normalizer = upper_cdf - lower_cdf

    def _build_instance(self, **new_params):
        """
        Build a new instance with the given parameters.

        Parameters
        ----------
        new_params : dict
            New parameters for the instance.

        Returns
        -------
        TruncatedDistribution
            The new instance.
        """
        distribution = new_params.pop("distribution")
        if new_params:
            raise ValueError(f"Unexpected arguments: {new_params}")
        return type(self)(distribution, self.lower_bound, self.upper_bound)

    def _valid_indices(self, x: np.ndarray):
        """
        Get valid indices within the bounds.

        Parameters
        ----------
        x : np.ndarray
            Data to check.

        Returns
        -------
        np.ndarray
            Boolean array of valid indices.
        """
        return (self.lower_bound <= x) & (x <= self.upper_bound)

    def _apply_constraints(self, x):
        """
        Apply constraints to the data.

        Parameters
        ----------
        x : array-like
            Data to apply constraints to.

        Returns
        -------
        array-like
            Data within the bounds.
        """
        return x[self._valid_indices(x)]

    def fit_instance(self, *args, **kwargs):
        """
        Fit the instance to the data.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        TruncatedDistribution
            The fitted instance.
        """
        kwargs.update(lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        return super().fit_instance(*args, **kwargs)

    def rvs(self, size: int, *args, **kwargs):
        """
        Generate random variates.

        Parameters
        ----------
        size : int
            Number of random variates to generate.
        args : tuple
            Positional arguments.
        kwargs : dict
            Keyword arguments.

        Returns
        -------
        np.ndarray
            Random variates.
        """
        u = Uniform(
            self.distribution.cdf(self.lower_bound),
            self.distribution.cdf(self.upper_bound),
        )
        return self.distribution.inverse_cdf(u.rvs(size, *args, **kwargs))

    def pdf(self, x: Obs):
        """
        Probability density function.

        Parameters
        ----------
        x : Obs
            Data to evaluate.

        Returns
        -------
        np.ndarray
            Probability density values.
        """
        return np.where(
            self._valid_indices(x),
            self.distribution.pdf(x) / self._normalizer,
            0.0,
        )

    def cdf(self, x: Obs):
        """
        Cumulative distribution function.

        Parameters
        ----------
        x : Obs
            Data to evaluate.

        Returns
        -------
        np.ndarray
            Cumulative distribution values.
        """
        right_range_x = (
            self.distribution.cdf(x) - self.distribution.cdf(self.lower_bound)
        ) / self._normalizer
        return np.where(self._valid_indices(x), right_range_x, 0.0)

    def isf(self, q: Obs):
        """
        Inverse survival function.

        Parameters
        ----------
        q : Obs
            Quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Inverse survival function values.
        """
        return self.distribution.isf(
            self.distribution.isf(self.upper_bound) + q * self._normalizer
        )

    def ppf(self, q: Obs):
        """
        Percent point function (inverse of cdf).

        Parameters
        ----------
        q : Obs
            Quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Percent point function values.
        """
        return self.distribution.ppf(
            self.distribution.cdf(self.lower_bound) + q * self._normalizer
        )
