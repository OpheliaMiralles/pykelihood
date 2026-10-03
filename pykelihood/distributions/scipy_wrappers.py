"""Opt-in structural distributions implemented with SciPy."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
import scipy
from packaging.version import Version
from scipy import stats
from scipy.special import xlog1py, xlogy
from scipy.stats import rv_continuous

from pykelihood.distributions.core import (
    Distribution,
    InvalidDistributionState,
    ParameterInput,
    ParameterState,
    RandomState,
)
from pykelihood.distributions.scipy_adapter import ParameterDefault, ScipyDistribution
from pykelihood.expr import Constant, Expr
from pykelihood.state import PositiveTransform


def _name_from_scipy_dist(scipy_dist: rv_continuous) -> str:
    return "".join(part.capitalize() for part in scipy_dist.name.split("_"))


class _WrappedScipyDistribution(ScipyDistribution):
    """Shared constructor for generated plain SciPy distributions."""

    _base_module: ClassVar[rv_continuous]
    __signature__: ClassVar[inspect.Signature]

    def __init__(self, *args: ParameterInput, **kwargs: ParameterInput) -> None:
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        super().__init__(
            self._base_module,
            bound.arguments,
            defaults={
                "loc": ParameterDefault(0.0),
                "scale": ParameterDefault(1.0, PositiveTransform()),
            },
        )


def wrap_scipy_distribution(
    scipy_dist: rv_continuous,
) -> type[_WrappedScipyDistribution]:
    """Create a structural distribution class for one SciPy continuous law.

    Shape parameters follow SciPy's native names and are required. ``loc`` and
    ``scale`` are optional free parameters initialized to 0 and 1 respectively.
    Literal arguments become constants through :class:`ScipyDistribution`.
    The generated constructor accepts arguments in SciPy's order: shape
    parameters, then ``loc`` and ``scale``.
    """
    shape_names = (
        ()
        if scipy_dist.shapes is None
        else tuple(name.strip() for name in scipy_dist.shapes.split(","))
    )
    parameter_names = (*shape_names, "loc", "scale")
    signature_parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if name in shape_names else None,
        )
        for name in parameter_names
    ]
    signature = inspect.Signature(signature_parameters)
    wrapper_name = _name_from_scipy_dist(scipy_dist)
    namespace: dict[str, Any] = {
        "_base_module": scipy_dist,
        "__doc__": f"Structural wrapper for ``scipy.stats.{scipy_dist.name}``.",
        "__module__": __name__,
        "__signature__": signature,
    }
    return type(wrapper_name, (_WrappedScipyDistribution,), namespace)


Alpha = wrap_scipy_distribution(stats.alpha)
Anglit = wrap_scipy_distribution(stats.anglit)
Arcsine = wrap_scipy_distribution(stats.arcsine)
Argus = wrap_scipy_distribution(stats.argus)
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
Normal = Norm
Gamma = wrap_scipy_distribution(stats.gamma)
Norminvgauss = wrap_scipy_distribution(stats.norminvgauss)
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
    DparetoLognorm = wrap_scipy_distribution(
        cast(rv_continuous, getattr(stats, "dpareto_lognorm"))  # noqa: B009
    )
    Landau = wrap_scipy_distribution(cast(rv_continuous, getattr(stats, "landau")))  # noqa: B009
    Irwinhall = wrap_scipy_distribution(
        cast(rv_continuous, getattr(stats, "irwinhall"))  # noqa: B009
    )


class Bernoulli(Distribution):
    """Bernoulli law with probability parameter ``p`` and integer samples."""

    def __init__(self, p: Expr | npt.ArrayLike) -> None:
        self._parameters = MappingProxyType(
            {"p": p if isinstance(p, Expr) else Constant(p)}
        )

    @property
    def p(self) -> Expr:
        return self._parameters["p"]

    @property
    def parameters(self) -> Mapping[str, Expr]:
        return self._parameters

    def _probability(self, state: ParameterState | None) -> npt.NDArray[np.float64]:
        probability = np.asarray(
            self.p.eval({} if state is None else state), dtype=np.float64
        )
        if np.any(~np.isfinite(probability)) or np.any(
            (probability < 0.0) | (probability > 1.0)
        ):
            raise InvalidDistributionState("Bernoulli probability must be in [0, 1].")
        return probability

    def log_prob(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        values = np.asarray(x)
        probability = self._probability(state)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = xlogy(values, probability) + xlog1py(1 - values, -probability)
        return np.asarray(
            np.where((values == 0) | (values == 1), scores, -np.inf), dtype=np.float64
        )

    def pmf(
        self, x: npt.ArrayLike, *, state: ParameterState | None = None
    ) -> npt.NDArray[np.float64]:
        return np.exp(self.log_prob(x, state=state))

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        *,
        state: ParameterState | None = None,
        random_state: RandomState = None,
    ) -> npt.NDArray[np.int64]:
        probability = self._probability(state)
        sample_shape = (
            () if size is None else (size,) if isinstance(size, int) else size
        )
        rng = (
            random_state
            if isinstance(random_state, (np.random.Generator, np.random.RandomState))
            else np.random.default_rng(random_state)
        )
        return np.asarray(
            rng.binomial(1, probability, size=sample_shape + probability.shape)
        )


__all__ = [
    "Alpha",
    "Anglit",
    "Arcsine",
    "Argus",
    "Bernoulli",
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
    "Gamma",
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
    "Loguniform",
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
    "wrap_scipy_distribution",
]

if Version(scipy.__version__) >= Version("1.15.0"):
    __all__.extend(["DparetoLognorm", "Irwinhall", "Landau"])
