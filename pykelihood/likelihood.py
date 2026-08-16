from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pykelihood.distributions.core import Distribution, ParameterState


def log_likelihood(
    model: Distribution, data: npt.ArrayLike, *, state: ParameterState | None = None
) -> float:
    """Return the summed log likelihood for ``data`` under ``model``."""
    return float(np.sum(model.logpdf(data, state=state)))


def negative_log_likelihood(
    model: Distribution, data: npt.ArrayLike, *, state: ParameterState | None = None
) -> float:
    """Return the scalar objective minimized for maximum likelihood fitting."""
    return -log_likelihood(model, data, state=state)
