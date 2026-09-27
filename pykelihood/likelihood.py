"""Likelihood objectives for structural distributions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pykelihood.distributions.core import Distribution, ParameterState


def log_likelihood(
    model: Distribution, data: npt.ArrayLike, *, state: ParameterState | None = None
) -> float:
    """Return the summed log likelihood for one model and one dataset.

    Pointwise scores must contain one value per observation. An additional
    model batch axis needs an explicit reduction policy and is not accepted.
    """
    observations = np.asarray(data)
    scores = np.asarray(model.log_prob(observations, state=state))
    expected_shape = () if observations.ndim == 0 else (len(observations),)
    if scores.shape != expected_shape:
        raise ValueError(
            f"Expected log_prob shape {expected_shape} for one dataset, "
            f"got {scores.shape}."
        )
    return float(np.sum(scores))


def negative_log_likelihood(
    model: Distribution, data: npt.ArrayLike, *, state: ParameterState | None = None
) -> float:
    """Return the scalar objective minimized for maximum likelihood fitting."""
    return -log_likelihood(model, data, state=state)
