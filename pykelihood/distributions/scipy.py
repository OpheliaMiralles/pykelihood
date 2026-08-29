from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Final, cast

from scipy import stats
from scipy.stats import rv_continuous

from pykelihood.distributions.base import Reparametrization, ScipyDistribution
from pykelihood.distributions.core import ParameterDefault, ParameterInput
from pykelihood.expr import Expr
from pykelihood.parameters import Parameter
from pykelihood.state import PositiveTransform


def _name_from_scipy_dist(scipy_dist: rv_continuous) -> str:
    """Generate a name for the distribution based on the scipy distribution class."""

    scipy_dist_name = type(scipy_dist).__name__.removesuffix("_gen")
    return "".join(map(str.capitalize, scipy_dist_name.split("_")))


def _native_defaults() -> dict[str, ParameterDefault]:
    return {
        "loc": ParameterDefault(0.0),
        "scale": ParameterDefault(1.0, PositiveTransform()),
    }


class _WrappedScipyDistribution(ScipyDistribution):
    """Constructor shared by the generated plain SciPy wrappers."""

    _base_module: rv_continuous

    def __init__(
        self,
        *args: ParameterInput,
        reparametrization: Reparametrization | None = None,
        **parameters: ParameterInput,
    ) -> None:
        if reparametrization is not None:
            if args:
                raise TypeError(
                    "Cannot use positional parameters with reparametrization."
                )
            if not parameters:
                raise TypeError(
                    "Reparametrized wrappers require at least one parameter."
                )
            compatibility_parameters: dict[str, Expr] = {}
            for name, value in parameters.items():
                if isinstance(value, Expr):
                    compatibility_parameters[name] = value
                elif value is None:
                    compatibility_parameters[name] = Parameter(init=0.0, name=name)
                else:
                    compatibility_parameters[name] = Parameter(init=value, name=name)
            super().__init__(
                self._base_module,
                compatibility_parameters,
                reparametrization=reparametrization,
            )
            return

        shape_names = (
            ()
            if self._base_module.shapes is None
            else tuple(self._base_module.shapes.split(", "))
        )
        names = ("loc", "scale", *shape_names)
        if len(args) > len(names):
            raise TypeError(f"Expected at most {len(names)} positional parameters.")

        values = dict(parameters)
        for name, value in zip(names, args):
            if name in values:
                raise TypeError(f"Parameter `{name}` was supplied more than once.")
            values[name] = value
        unknown = set(values) - set(names)
        if unknown:
            unknown_names = ", ".join(sorted(unknown))
            raise TypeError(f"Unexpected parameter(s): {unknown_names}.")
        for name in names[2:]:
            if values.get(name) is None:
                raise TypeError(f"Missing required distribution parameter: {name}")

        super().__init__(
            self._base_module,
            {name: values.get(name) for name in names},
            defaults=_native_defaults(),
        )


def wrap_scipy_distribution(
    scipy_dist: rv_continuous, *, name: str | None = None
) -> type[_WrappedScipyDistribution]:
    """Wrap a scipy distribution class to create a ScipyDistribution subclass."""

    scipy_dist_name = type(scipy_dist).__name__.removesuffix("_gen")
    clean_dist_name = name or _name_from_scipy_dist(scipy_dist)
    dist_params_names = ("loc", "scale") + tuple(
        scipy_dist.shapes.split(", ") if scipy_dist.shapes else ()
    )

    docstring = f"""\\
    {clean_dist_name} distribution.

    Parameters
    ----------
    loc : float, optional
        Location parameter, by default 0.0.
    scale : float, optional
        Scale parameter, by default 1.0.\\
    """

    def format_param_docstring(param: str) -> str:
        return f"""
    {param} : float, mandatory
        Shape parameter. See the SciPy documentation for the {scipy_dist_name} distribution for details.\\
        """

    for param in dist_params_names[2:]:
        docstring += format_param_docstring(param)

    return cast(
        type[_WrappedScipyDistribution],
        type(
            clean_dist_name,
            (_WrappedScipyDistribution,),
            {
                "_base_module": scipy_dist,
                "__doc__": docstring,
                "__module__": wrap_scipy_distribution.__module__,
            },
        ),
    )


_SCIPY_WRAPPER_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("Alpha", "alpha"),
    ("Anglit", "anglit"),
    ("Arcsine", "arcsine"),
    ("Argus", "argus"),
    ("Beta", "beta"),
    ("Betaprime", "betaprime"),
    ("Bradford", "bradford"),
    ("Burr", "burr"),
    ("Burr12", "burr12"),
    ("Cauchy", "cauchy"),
    ("Chi", "chi"),
    ("Chi2", "chi2"),
    ("Cosine", "cosine"),
    ("Crystalball", "crystalball"),
    ("Dgamma", "dgamma"),
    ("Dweibull", "dweibull"),
    ("Erlang", "erlang"),
    ("Expon", "expon"),
    ("Exponnorm", "exponnorm"),
    ("Exponpow", "exponpow"),
    ("Exponweib", "exponweib"),
    ("F", "f"),
    ("Fatiguelife", "fatiguelife"),
    ("Fisk", "fisk"),
    ("Foldcauchy", "foldcauchy"),
    ("Foldnorm", "foldnorm"),
    ("Gamma", "gamma"),
    ("Gausshyper", "gausshyper"),
    ("Genexpon", "genexpon"),
    ("Genextreme", "genextreme"),
    ("Gengamma", "gengamma"),
    ("Genhalflogistic", "genhalflogistic"),
    ("Genhyperbolic", "genhyperbolic"),
    ("Geninvgauss", "geninvgauss"),
    ("Genlogistic", "genlogistic"),
    ("Gennorm", "gennorm"),
    ("Genpareto", "genpareto"),
    ("Gibrat", "gibrat"),
    ("Gompertz", "gompertz"),
    ("GumbelL", "gumbel_l"),
    ("GumbelR", "gumbel_r"),
    ("Halfcauchy", "halfcauchy"),
    ("Halfgennorm", "halfgennorm"),
    ("Halflogistic", "halflogistic"),
    ("Halfnorm", "halfnorm"),
    ("Hypsecant", "hypsecant"),
    ("Invgamma", "invgamma"),
    ("Invgauss", "invgauss"),
    ("Invweibull", "invweibull"),
    ("JfSkewT", "jf_skew_t"),
    ("Johnsonsb", "johnsonsb"),
    ("Johnsonsu", "johnsonsu"),
    ("Kappa3", "kappa3"),
    ("Kappa4", "kappa4"),
    ("Ksone", "ksone"),
    ("Kstwo", "kstwo"),
    ("Kstwobign", "kstwobign"),
    ("Laplace", "laplace"),
    ("LaplaceAsymmetric", "laplace_asymmetric"),
    ("Levy", "levy"),
    ("LevyL", "levy_l"),
    ("LevyStable", "levy_stable"),
    ("Loggamma", "loggamma"),
    ("Logistic", "logistic"),
    ("Loglaplace", "loglaplace"),
    ("Lognorm", "lognorm"),
    ("Lomax", "lomax"),
    ("Maxwell", "maxwell"),
    ("Mielke", "mielke"),
    ("Moyal", "moyal"),
    ("Nakagami", "nakagami"),
    ("Ncf", "ncf"),
    ("Nct", "nct"),
    ("Ncx2", "ncx2"),
    ("Norm", "norm"),
    ("Norminvgauss", "norminvgauss"),
    ("Pareto", "pareto"),
    ("Pearson3", "pearson3"),
    ("Powerlaw", "powerlaw"),
    ("Powerlognorm", "powerlognorm"),
    ("Powernorm", "powernorm"),
    ("Rayleigh", "rayleigh"),
    ("Rdist", "rdist"),
    ("Recipinvgauss", "recipinvgauss"),
    ("Reciprocal", "reciprocal"),
    ("RelBreitwigner", "rel_breitwigner"),
    ("Rice", "rice"),
    ("Semicircular", "semicircular"),
    ("Skewcauchy", "skewcauchy"),
    ("Skewnorm", "skewnorm"),
    ("StudentizedRange", "studentized_range"),
    ("T", "t"),
    ("Trapezoid", "trapezoid"),
    ("Triang", "triang"),
    ("Truncexpon", "truncexpon"),
    ("Truncnorm", "truncnorm"),
    ("Truncpareto", "truncpareto"),
    ("TruncweibullMin", "truncweibull_min"),
    ("Tukeylambda", "tukeylambda"),
    ("Uniform", "uniform"),
    ("Vonmises", "vonmises"),
    ("Wald", "wald"),
    ("WeibullMax", "weibull_max"),
    ("WeibullMin", "weibull_min"),
    ("Wrapcauchy", "wrapcauchy"),
    ("DparetoLognorm", "dpareto_lognorm"),
    ("Landau", "landau"),
    ("Irwinhall", "irwinhall"),
)

_SCIPY_ALIASES: Final[tuple[tuple[str, str], ...]] = (
    ("Normal", "Norm"),
    ("Loguniform", "Reciprocal"),
    ("Trapz", "Trapezoid"),
    ("VonmisesLine", "Vonmises"),
)

_HELPER_EXPORTS: Final[tuple[str, ...]] = (
    "_name_from_scipy_dist",
    "wrap_scipy_distribution",
)


def _get_scipy_distribution(name: str) -> rv_continuous | None:
    candidates = ("trapz",) if name == "trapezoid" else ()
    for candidate_name in (name, *candidates):
        candidate = getattr(stats, candidate_name, None)
        if isinstance(candidate, rv_continuous):
            return candidate
    return None


def _register_scipy_wrappers(namespace: MutableMapping[str, object]) -> tuple[str, ...]:
    for public_name, scipy_name in _SCIPY_WRAPPER_SPECS:
        scipy_dist = _get_scipy_distribution(scipy_name)
        if scipy_dist is not None:
            namespace[public_name] = wrap_scipy_distribution(
                scipy_dist, name="Trapezoid" if scipy_name == "trapezoid" else None
            )

    for alias_name, target_name in _SCIPY_ALIASES:
        target = namespace.get(target_name)
        if isinstance(target, type) and issubclass(target, ScipyDistribution):
            namespace[alias_name] = target

    return tuple(
        name
        for public_name, _ in _SCIPY_WRAPPER_SPECS
        for name in (
            public_name,
            *(alias for alias, target in _SCIPY_ALIASES if target == public_name),
        )
        if name in namespace
    )


_REGISTERED_NAMES = _register_scipy_wrappers(
    cast(MutableMapping[str, object], globals())
)

if TYPE_CHECKING:
    Beta: type[_WrappedScipyDistribution]
    Gamma: type[_WrappedScipyDistribution]
    Loguniform: type[_WrappedScipyDistribution]
    Norm: type[_WrappedScipyDistribution]
    Normal: type[_WrappedScipyDistribution]
    Pareto: type[_WrappedScipyDistribution]
    Reciprocal: type[_WrappedScipyDistribution]
    Trapz: type[_WrappedScipyDistribution]
    Trapezoid: type[_WrappedScipyDistribution]
    Uniform: type[_WrappedScipyDistribution]
    Vonmises: type[_WrappedScipyDistribution]
    VonmisesLine: type[_WrappedScipyDistribution]
    __all__: list[str]
else:
    __all__ = [*_HELPER_EXPORTS, *_REGISTERED_NAMES]
