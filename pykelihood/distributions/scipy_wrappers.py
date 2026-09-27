"""Opt-in structural wrappers for selected SciPy continuous distributions."""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

from scipy import stats
from scipy.stats import rv_continuous

from pykelihood.distributions.core import (
    ParameterDefault,
    ParameterInput,
    ScipyDistribution,
)
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


Norm = wrap_scipy_distribution(stats.norm)
Normal = Norm
Gamma = wrap_scipy_distribution(stats.gamma)
Genextreme = wrap_scipy_distribution(stats.genextreme)

__all__ = ["Gamma", "Genextreme", "Norm", "Normal", "wrap_scipy_distribution"]
