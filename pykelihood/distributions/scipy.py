from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import scipy
from numpy.typing import ArrayLike
from packaging.version import Version
from scipy import stats

from pykelihood.distributions.base import ScipyDistribution

if TYPE_CHECKING:
    from typing import Self


def _name_from_scipy_dist(scipy_dist: stats.rv_continuous) -> str:
    """Generate a name for the distribution based on the scipy distribution class."""
    scipy_dist_name = type(scipy_dist).__name__.removesuffix("_gen")
    return "".join(map(str.capitalize, scipy_dist_name.split("_")))


class Reparametrization(Protocol):
    def __call__(self, params: dict[str, ArrayLike]) -> dict[str, ArrayLike]: ...


def wrap_scipy_distribution(scipy_dist: stats.rv_continuous) -> type[ScipyDistribution]:
    """Wrap a scipy distribution class to create a ScipyDistribution subclass."""
    scipy_dist_name = type(scipy_dist).__name__.removesuffix("_gen")
    clean_dist_name = _name_from_scipy_dist(scipy_dist)
    dist_params_names = ("loc", "scale") + tuple(
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

    for param in dist_params_names[2:]:
        docstring += format_param_docstring(param)

    class Wrapper(ScipyDistribution):
        _base_module = scipy_dist
        __doc__ = docstring

        def __init__(
            self, *, reparametrization: Reparametrization | None = None, **params
        ):
            self.reparametrization = reparametrization or (lambda p: p)
            if reparametrization is None:
                self._params_names = dist_params_names
                assert self._params_names[:2] == ("loc", "scale")
                shape_args = self._params_names[2:]
                for arg in shape_args:
                    if arg not in params:
                        raise ValueError(
                            f"Missing shape parameter `{arg}` when initializing {type(self).__name__} distribution."
                        )
                params = {
                    "loc": params.get("loc", 0.0),
                    "scale": params.get("scale", 1.0),
                } | {a: params[a] for a in shape_args}
            else:
                self._params_names = tuple(params)
            super().__init__(*params.values())

        def _build_instance(self, **params) -> Self:
            return type(self)(reparametrization=self.reparametrization, **params)

        @property
        def params_names(self) -> tuple[str, ...]:
            """Return the names of the parameters."""
            return self._params_names

        def _to_scipy_args(self, **kwargs):
            values = {k: kwargs.get(k, getattr(self, k)()) for k in self.params_names}
            return self.reparametrization(values)

    Wrapper.__name__ = clean_dist_name
    Wrapper.__qualname__ = f"{Wrapper.__module__}.{Wrapper.__name__}"
    return Wrapper


Alpha = wrap_scipy_distribution(stats.alpha)
Anglit = wrap_scipy_distribution(stats.anglit)
Arcsine = wrap_scipy_distribution(stats.arcsine)
Argus = wrap_scipy_distribution(stats.argus)
# Beta = wrap_scipy_distribution(stats.beta)
Betaprime = wrap_scipy_distribution(stats.betaprime)
Bradford = wrap_scipy_distribution(stats.bradford)
Burr = wrap_scipy_distribution(stats.burr)
Burr12 = wrap_scipy_distribution(stats.burr12)
Cauchy = wrap_scipy_distribution(stats.cauchy)
Chi = wrap_scipy_distribution(stats.chi)
Chi2 = wrap_scipy_distribution(stats.chi2)
Cosine = wrap_scipy_distribution(stats.cosine)
Crystalball = wrap_scipy_distribution(stats.crystalball)
Dgamma = wrap_scipy_distribution(stats.dgamma)
Dweibull = wrap_scipy_distribution(stats.dweibull)
Erlang = wrap_scipy_distribution(stats.erlang)
Expon = wrap_scipy_distribution(stats.expon)
Exponnorm = wrap_scipy_distribution(stats.exponnorm)
Exponpow = wrap_scipy_distribution(stats.exponpow)
Exponweib = wrap_scipy_distribution(stats.exponweib)
F = wrap_scipy_distribution(stats.f)
Fatiguelife = wrap_scipy_distribution(stats.fatiguelife)
Fisk = wrap_scipy_distribution(stats.fisk)
Foldcauchy = wrap_scipy_distribution(stats.foldcauchy)
Foldnorm = wrap_scipy_distribution(stats.foldnorm)
# Gamma = wrap_scipy_distribution(stats.gamma)
Gausshyper = wrap_scipy_distribution(stats.gausshyper)
Genexpon = wrap_scipy_distribution(stats.genexpon)
Genextreme = wrap_scipy_distribution(stats.genextreme)
Gengamma = wrap_scipy_distribution(stats.gengamma)
Genhalflogistic = wrap_scipy_distribution(stats.genhalflogistic)
Genhyperbolic = wrap_scipy_distribution(stats.genhyperbolic)
Geninvgauss = wrap_scipy_distribution(stats.geninvgauss)
Genlogistic = wrap_scipy_distribution(stats.genlogistic)
Gennorm = wrap_scipy_distribution(stats.gennorm)
Genpareto = wrap_scipy_distribution(stats.genpareto)
Gibrat = wrap_scipy_distribution(stats.gibrat)
Gompertz = wrap_scipy_distribution(stats.gompertz)
GumbelL = wrap_scipy_distribution(stats.gumbel_l)
GumbelR = wrap_scipy_distribution(stats.gumbel_r)
Halfcauchy = wrap_scipy_distribution(stats.halfcauchy)
Halfgennorm = wrap_scipy_distribution(stats.halfgennorm)
Halflogistic = wrap_scipy_distribution(stats.halflogistic)
Halfnorm = wrap_scipy_distribution(stats.halfnorm)
Hypsecant = wrap_scipy_distribution(stats.hypsecant)
Invgamma = wrap_scipy_distribution(stats.invgamma)
Invgauss = wrap_scipy_distribution(stats.invgauss)
Invweibull = wrap_scipy_distribution(stats.invweibull)
JfSkewT = wrap_scipy_distribution(stats.jf_skew_t)
Johnsonsb = wrap_scipy_distribution(stats.johnsonsb)
Johnsonsu = wrap_scipy_distribution(stats.johnsonsu)
Kappa3 = wrap_scipy_distribution(stats.kappa3)
Kappa4 = wrap_scipy_distribution(stats.kappa4)
Ksone = wrap_scipy_distribution(stats.ksone)
Kstwo = wrap_scipy_distribution(stats.kstwo)
Kstwobign = wrap_scipy_distribution(stats.kstwobign)
Laplace = wrap_scipy_distribution(stats.laplace)
LaplaceAsymmetric = wrap_scipy_distribution(stats.laplace_asymmetric)
Levy = wrap_scipy_distribution(stats.levy)
LevyL = wrap_scipy_distribution(stats.levy_l)
LevyStable = wrap_scipy_distribution(stats.levy_stable)
Loggamma = wrap_scipy_distribution(stats.loggamma)
Logistic = wrap_scipy_distribution(stats.logistic)
Loglaplace = wrap_scipy_distribution(stats.loglaplace)
Lognorm = wrap_scipy_distribution(stats.lognorm)
Lomax = wrap_scipy_distribution(stats.lomax)
Maxwell = wrap_scipy_distribution(stats.maxwell)
Mielke = wrap_scipy_distribution(stats.mielke)
Moyal = wrap_scipy_distribution(stats.moyal)
Nakagami = wrap_scipy_distribution(stats.nakagami)
Ncf = wrap_scipy_distribution(stats.ncf)
Nct = wrap_scipy_distribution(stats.nct)
Ncx2 = wrap_scipy_distribution(stats.ncx2)
Norm = wrap_scipy_distribution(stats.norm)
Normal = Norm  # alias for backward compatibility
Norminvgauss = wrap_scipy_distribution(stats.norminvgauss)
# Pareto = wrap_scipy_distribution(stats.pareto)
Pearson3 = wrap_scipy_distribution(stats.pearson3)
Powerlaw = wrap_scipy_distribution(stats.powerlaw)
Powerlognorm = wrap_scipy_distribution(stats.powerlognorm)
Powernorm = wrap_scipy_distribution(stats.powernorm)
Rayleigh = wrap_scipy_distribution(stats.rayleigh)
Rdist = wrap_scipy_distribution(stats.rdist)
Recipinvgauss = wrap_scipy_distribution(stats.recipinvgauss)
Loguniform = wrap_scipy_distribution(stats.loguniform)
Reciprocal = wrap_scipy_distribution(stats.reciprocal)
RelBreitwigner = wrap_scipy_distribution(stats.rel_breitwigner)
Rice = wrap_scipy_distribution(stats.rice)
Semicircular = wrap_scipy_distribution(stats.semicircular)
Skewcauchy = wrap_scipy_distribution(stats.skewcauchy)
Skewnorm = wrap_scipy_distribution(stats.skewnorm)
StudentizedRange = wrap_scipy_distribution(stats.studentized_range)
T = wrap_scipy_distribution(stats.t)
Trapezoid = wrap_scipy_distribution(stats.trapezoid)
Trapz = Trapezoid
Triang = wrap_scipy_distribution(stats.triang)
Truncexpon = wrap_scipy_distribution(stats.truncexpon)
Truncnorm = wrap_scipy_distribution(stats.truncnorm)
Truncpareto = wrap_scipy_distribution(stats.truncpareto)
TruncweibullMin = wrap_scipy_distribution(stats.truncweibull_min)
Tukeylambda = wrap_scipy_distribution(stats.tukeylambda)
Uniform = wrap_scipy_distribution(stats.uniform)
Vonmises = wrap_scipy_distribution(stats.vonmises)
VonmisesLine = wrap_scipy_distribution(stats.vonmises_line)
Wald = wrap_scipy_distribution(stats.wald)
WeibullMax = wrap_scipy_distribution(stats.weibull_max)
WeibullMin = wrap_scipy_distribution(stats.weibull_min)
Wrapcauchy = wrap_scipy_distribution(stats.wrapcauchy)

if Version(scipy.__version__) >= Version("1.15.0"):
    DparetoLognorm = wrap_scipy_distribution(stats.dpareto_lognorm)
    Landau = wrap_scipy_distribution(stats.landau)
    Irwinhall = wrap_scipy_distribution(stats.irwinhall)

__all__ = [
    "_name_from_scipy_dist",
    "wrap_scipy_distribution",
    "Alpha",
    "Anglit",
    "Arcsine",
    "Argus",
    "Betaprime",
    "Bradford",
    "Burr",
    "Burr12",
    "Cauchy",
    "Chi",
    "Chi2",
    "Cosine",
    "Crystalball",
    "Dgamma",
    "Dweibull",
    "Erlang",
    "Expon",
    "Exponnorm",
    "Exponpow",
    "Exponweib",
    "F",
    "Fatiguelife",
    "Fisk",
    "Foldcauchy",
    "Foldnorm",
    "Gausshyper",
    "Genexpon",
    "Genextreme",
    "Gengamma",
    "Genhalflogistic",
    "Genhyperbolic",
    "Geninvgauss",
    "Genlogistic",
    "Gennorm",
    "Genpareto",
    "Gibrat",
    "Gompertz",
    "GumbelL",
    "GumbelR",
    "Halfcauchy",
    "Halfgennorm",
    "Halflogistic",
    "Halfnorm",
    "Hypsecant",
    "Invgamma",
    "Invgauss",
    "Invweibull",
    "JfSkewT",
    "Johnsonsb",
    "Johnsonsu",
    "Kappa3",
    "Kappa4",
    "Ksone",
    "Kstwo",
    "Kstwobign",
    "Laplace",
    "LaplaceAsymmetric",
    "Levy",
    "LevyL",
    "LevyStable",
    "Loggamma",
    "Logistic",
    "Loglaplace",
    "Lognorm",
    "Lomax",
    "Maxwell",
    "Mielke",
    "Moyal",
    "Nakagami",
    "Ncf",
    "Nct",
    "Ncx2",
    "Norm",
    "Normal",
    "Norminvgauss",
    "Pearson3",
    "Powerlaw",
    "Powerlognorm",
    "Powernorm",
    "Rayleigh",
    "Rdist",
    "Recipinvgauss",
    "Loguniform",
    "Reciprocal",
    "RelBreitwigner",
    "Rice",
    "Semicircular",
    "Skewcauchy",
    "Skewnorm",
    "StudentizedRange",
    "T",
    "Trapezoid",
    "Trapz",
    "Triang",
    "Truncexpon",
    "Truncnorm",
    "Truncpareto",
    "TruncweibullMin",
    "Tukeylambda",
    "Uniform",
    "Vonmises",
    "VonmisesLine",
    "Wald",
    "WeibullMax",
    "WeibullMin",
    "Wrapcauchy",
]

if Version(scipy.__version__) >= Version("1.15.0"):
    __all__.extend(["DparetoLognorm", "Landau", "Irwinhall"])
