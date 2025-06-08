from __future__ import annotations

import scipy.special
from packaging.version import Version
from scipy import stats

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
