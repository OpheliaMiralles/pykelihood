"""Information criteria for explicit-state maximum-likelihood fits."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pykelihood.likelihood import log_likelihood
from pykelihood.parametric.fitting import FitResult
from pykelihood.state import ParameterLayout


def _fit_summary(fit: FitResult, data: npt.ArrayLike) -> tuple[float, int]:
    if not fit.optimize_result.success or not np.isfinite(fit.optimize_result.fun):
        raise ValueError("Information criteria require a successful finite fit.")
    log_score = log_likelihood(fit.model, data, state=fit.state)
    free_count = ParameterLayout.from_expr(fit.model).without(fit.fixed).vector_size
    return log_score, free_count


def aic(fit: FitResult, data: npt.ArrayLike) -> float:
    """Akaike information criterion using the fit's free coordinates."""
    log_score, free_count = _fit_summary(fit, data)
    return float(2 * free_count - 2 * log_score)


def bic(fit: FitResult, data: npt.ArrayLike) -> float:
    """Bayesian information criterion using the fit's free coordinates."""
    observations = np.asarray(data)
    count = 1 if observations.ndim == 0 else len(observations)
    if count == 0:
        raise ValueError("BIC requires at least one observation.")
    log_score, free_count = _fit_summary(fit, observations)
    return float(np.log(count) * free_count - 2 * log_score)
