from typing import TYPE_CHECKING

from pykelihood.distributions import custom, scipy
from pykelihood.distributions.base import Distribution, ScipyDistribution
from pykelihood.distributions.custom import *  # noqa: F403
from pykelihood.distributions.scipy import *  # noqa: F403

if TYPE_CHECKING:
    from pykelihood.distributions.scipy import (
        _name_from_scipy_dist as _name_from_scipy_dist,
    )

    __all__: list[str]
else:
    __all__ = [*scipy.__all__, *custom.__all__, "Distribution", "ScipyDistribution"]
