"""Post-fit information criteria for explicit-state maximum-likelihood fits."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from pykelihood.distributions.core import Distribution
from pykelihood.inference.mle import FitResult
from pykelihood.likelihood import log_likelihood
from pykelihood.state import ParameterLayout

_ModelT = TypeVar("_ModelT", bound=Distribution)


def _fit_summary(fit: FitResult[_ModelT]) -> tuple[float, int]:
    if not fit.optimize_result.success or not np.isfinite(fit.optimize_result.fun):
        raise ValueError("Information criteria require a successful finite fit.")
    log_score = log_likelihood(fit.model, fit.data, state=fit.state)
    if not np.isfinite(log_score):
        raise ValueError(
            "Information criteria require a finite likelihood for the data."
        )
    free_count = ParameterLayout.from_node(fit.model).without(fit.fixed).vector_size
    return log_score, free_count


def aic(fit: FitResult[_ModelT]) -> float:
    """Akaike information criterion on the dataset used to obtain ``fit``."""
    log_score, free_count = _fit_summary(fit)
    return float(2 * free_count - 2 * log_score)


def bic(fit: FitResult[_ModelT]) -> float:
    """Bayesian information criterion on the dataset used to obtain ``fit``."""
    observations = fit.data
    count = 1 if observations.ndim == 0 else len(observations)
    if count == 0:
        raise ValueError("BIC requires at least one observation.")
    log_score, free_count = _fit_summary(fit)
    return float(np.log(count) * free_count - 2 * log_score)
