from __future__ import annotations

import scipy.special
from packaging.version import Version
from scipy import stats
from scipy import stats as _stats

from pykelihood.distributions import ScipyDistribution


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
            "_base_module": scipy_dist,
            "params_names": params_names,
            "__init__": __init__,
            "_to_scipy_args": _to_scipy_args,
            "__doc__": docstring,
        },
    )


Alpha = _wrap_scipy_distribution(_stats.alpha)
Anglit = _wrap_scipy_distribution(_stats.anglit)
Arcsine = _wrap_scipy_distribution(_stats.arcsine)
Argus = _wrap_scipy_distribution(_stats.argus)
# Beta = _wrap_scipy_distribution(_stats.beta)
Betaprime = _wrap_scipy_distribution(_stats.betaprime)
Bradford = _wrap_scipy_distribution(_stats.bradford)
Burr = _wrap_scipy_distribution(_stats.burr)
Burr12 = _wrap_scipy_distribution(_stats.burr12)
Cauchy = _wrap_scipy_distribution(_stats.cauchy)
Chi = _wrap_scipy_distribution(_stats.chi)
Chi2 = _wrap_scipy_distribution(_stats.chi2)
Cosine = _wrap_scipy_distribution(_stats.cosine)
Crystalball = _wrap_scipy_distribution(_stats.crystalball)
Dgamma = _wrap_scipy_distribution(_stats.dgamma)
Dweibull = _wrap_scipy_distribution(_stats.dweibull)
Erlang = _wrap_scipy_distribution(_stats.erlang)
Expon = _wrap_scipy_distribution(_stats.expon)
Exponnorm = _wrap_scipy_distribution(_stats.exponnorm)
Exponpow = _wrap_scipy_distribution(_stats.exponpow)
Exponweib = _wrap_scipy_distribution(_stats.exponweib)
F = _wrap_scipy_distribution(_stats.f)
Fatiguelife = _wrap_scipy_distribution(_stats.fatiguelife)
Fisk = _wrap_scipy_distribution(_stats.fisk)
Foldcauchy = _wrap_scipy_distribution(_stats.foldcauchy)
Foldnorm = _wrap_scipy_distribution(_stats.foldnorm)
# Gamma = _wrap_scipy_distribution(_stats.gamma)
Gausshyper = _wrap_scipy_distribution(_stats.gausshyper)
Genexpon = _wrap_scipy_distribution(_stats.genexpon)
Genextreme = _wrap_scipy_distribution(_stats.genextreme)
Gengamma = _wrap_scipy_distribution(_stats.gengamma)
Genhalflogistic = _wrap_scipy_distribution(_stats.genhalflogistic)
Genhyperbolic = _wrap_scipy_distribution(_stats.genhyperbolic)
Geninvgauss = _wrap_scipy_distribution(_stats.geninvgauss)
Genlogistic = _wrap_scipy_distribution(_stats.genlogistic)
Gennorm = _wrap_scipy_distribution(_stats.gennorm)
Genpareto = _wrap_scipy_distribution(_stats.genpareto)
Gibrat = _wrap_scipy_distribution(_stats.gibrat)
Gompertz = _wrap_scipy_distribution(_stats.gompertz)
GumbelL = _wrap_scipy_distribution(_stats.gumbel_l)
GumbelR = _wrap_scipy_distribution(_stats.gumbel_r)
Halfcauchy = _wrap_scipy_distribution(_stats.halfcauchy)
Halfgennorm = _wrap_scipy_distribution(_stats.halfgennorm)
Halflogistic = _wrap_scipy_distribution(_stats.halflogistic)
Halfnorm = _wrap_scipy_distribution(_stats.halfnorm)
Hypsecant = _wrap_scipy_distribution(_stats.hypsecant)
Invgamma = _wrap_scipy_distribution(_stats.invgamma)
Invgauss = _wrap_scipy_distribution(_stats.invgauss)
Invweibull = _wrap_scipy_distribution(_stats.invweibull)
JfSkewT = _wrap_scipy_distribution(_stats.jf_skew_t)
Johnsonsb = _wrap_scipy_distribution(_stats.johnsonsb)
Johnsonsu = _wrap_scipy_distribution(_stats.johnsonsu)
Kappa3 = _wrap_scipy_distribution(_stats.kappa3)
Kappa4 = _wrap_scipy_distribution(_stats.kappa4)
Ksone = _wrap_scipy_distribution(_stats.ksone)
Kstwo = _wrap_scipy_distribution(_stats.kstwo)
Kstwobign = _wrap_scipy_distribution(_stats.kstwobign)
Laplace = _wrap_scipy_distribution(_stats.laplace)
LaplaceAsymmetric = _wrap_scipy_distribution(_stats.laplace_asymmetric)
Levy = _wrap_scipy_distribution(_stats.levy)
LevyL = _wrap_scipy_distribution(_stats.levy_l)
LevyStable = _wrap_scipy_distribution(_stats.levy_stable)
Loggamma = _wrap_scipy_distribution(_stats.loggamma)
Logistic = _wrap_scipy_distribution(_stats.logistic)
Loglaplace = _wrap_scipy_distribution(_stats.loglaplace)
Lognorm = _wrap_scipy_distribution(_stats.lognorm)
Lomax = _wrap_scipy_distribution(_stats.lomax)
Maxwell = _wrap_scipy_distribution(_stats.maxwell)
Mielke = _wrap_scipy_distribution(_stats.mielke)
Moyal = _wrap_scipy_distribution(_stats.moyal)
Nakagami = _wrap_scipy_distribution(_stats.nakagami)
Ncf = _wrap_scipy_distribution(_stats.ncf)
Nct = _wrap_scipy_distribution(_stats.nct)
Ncx2 = _wrap_scipy_distribution(_stats.ncx2)
Norm = _wrap_scipy_distribution(_stats.norm)
Normal = Norm  # alias for backward compatibility
Norminvgauss = _wrap_scipy_distribution(_stats.norminvgauss)
# Pareto = _wrap_scipy_distribution(_stats.pareto)
Pearson3 = _wrap_scipy_distribution(_stats.pearson3)
Powerlaw = _wrap_scipy_distribution(_stats.powerlaw)
Powerlognorm = _wrap_scipy_distribution(_stats.powerlognorm)
Powernorm = _wrap_scipy_distribution(_stats.powernorm)
Rayleigh = _wrap_scipy_distribution(_stats.rayleigh)
Rdist = _wrap_scipy_distribution(_stats.rdist)
Recipinvgauss = _wrap_scipy_distribution(_stats.recipinvgauss)
Reciprocal = _wrap_scipy_distribution(_stats.loguniform)
Reciprocal = _wrap_scipy_distribution(_stats.reciprocal)
RelBreitwigner = _wrap_scipy_distribution(_stats.rel_breitwigner)
Rice = _wrap_scipy_distribution(_stats.rice)
Semicircular = _wrap_scipy_distribution(_stats.semicircular)
Skewcauchy = _wrap_scipy_distribution(_stats.skewcauchy)
Skewnorm = _wrap_scipy_distribution(_stats.skewnorm)
StudentizedRange = _wrap_scipy_distribution(_stats.studentized_range)
T = _wrap_scipy_distribution(_stats.t)
Trapezoid = _wrap_scipy_distribution(_stats.trapezoid)
Trapz = Trapezoid
Triang = _wrap_scipy_distribution(_stats.triang)
Truncexpon = _wrap_scipy_distribution(_stats.truncexpon)
Truncnorm = _wrap_scipy_distribution(_stats.truncnorm)
Truncpareto = _wrap_scipy_distribution(_stats.truncpareto)
TruncweibullMin = _wrap_scipy_distribution(_stats.truncweibull_min)
Tukeylambda = _wrap_scipy_distribution(_stats.tukeylambda)
Uniform = _wrap_scipy_distribution(_stats.uniform)
Vonmises = _wrap_scipy_distribution(_stats.vonmises_line)
Vonmises = _wrap_scipy_distribution(_stats.vonmises)
Wald = _wrap_scipy_distribution(_stats.wald)
WeibullMax = _wrap_scipy_distribution(_stats.weibull_max)
WeibullMin = _wrap_scipy_distribution(_stats.weibull_min)
Wrapcauchy = _wrap_scipy_distribution(_stats.wrapcauchy)

if Version(scipy.__version__) >= Version("1.15.0"):
    DparetoLognorm = _wrap_scipy_distribution(_stats.dpareto_lognorm)
    Landau = _wrap_scipy_distribution(_stats.landau)
    Irwinhall = _wrap_scipy_distribution(_stats.irwinhall)
