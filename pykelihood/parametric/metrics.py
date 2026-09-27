"""Predictive scores for explicit-state univariate continuous distributions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from pykelihood.distributions.core import (
    ParameterState,
    UnivariateContinuousDistribution,
)


def _observations(data: npt.ArrayLike) -> npt.NDArray[np.float64]:
    observations = np.asarray(data, dtype=np.float64)
    if observations.size == 0 or not np.all(np.isfinite(observations)):
        raise ValueError("Scores require finite, nonempty observations.")
    return observations


def _prediction_for_observations(
    prediction: npt.ArrayLike, observations: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape not in ((), observations.shape):
        raise ValueError(
            f"Expected a scalar prediction or shape {observations.shape}, "
            f"got {values.shape}."
        )
    return values


def brier_score(
    model: UnivariateContinuousDistribution,
    data: npt.ArrayLike,
    threshold: float,
    *,
    state: ParameterState | None = None,
) -> float:
    """Mean squared error for the forecast event ``X >= threshold``."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    observations = _observations(data)
    exceedance = 1.0 - _prediction_for_observations(
        model.cdf(threshold, state=state), observations
    )
    return float(np.mean((exceedance - (observations >= threshold)) ** 2))


def quantile_score(
    model: UnivariateContinuousDistribution,
    data: npt.ArrayLike,
    quantile: float,
    *,
    state: ParameterState | None = None,
) -> float:
    """Mean pinball loss at a probability strictly between zero and one."""
    if not np.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1, excluding endpoints.")
    observations = _observations(data)
    forecast = _prediction_for_observations(
        model.ppf(quantile, state=state), observations
    )
    residual = observations - forecast
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def crps(
    model: UnivariateContinuousDistribution,
    data: npt.ArrayLike,
    *,
    state: ParameterState | None = None,
) -> float:
    """Mean continuous ranked probability score, one forecast per observation."""
    observations = _observations(data)

    def squared_cdf_error(value: float) -> float:
        forecast = _prediction_for_observations(
            model.cdf(value, state=state), observations
        )
        return float(np.mean((forecast - (observations <= value)) ** 2))

    values = np.unique(observations)
    left, _ = quad(squared_cdf_error, -np.inf, values[0])
    right, _ = quad(squared_cdf_error, values[-1], np.inf)
    middle = 0.0
    if len(values) > 1:
        middle, _ = quad(
            squared_cdf_error,
            values[0],
            values[-1],
            points=values[1:-1],
            limit=len(values) + 50,
        )
    return float(left + middle + right)
