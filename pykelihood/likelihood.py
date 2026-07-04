from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pykelihood.distributions.base import Distribution

def log_likelihood(model: Distribution, data) -> float:
    """Compute the log-likelihood of data under the model."""
    return float(np.sum(model.logpdf(data)))

def negative_log_likelihood(model: Distribution, data) -> float:
    """Compute the negative log-likelihood of data under the model."""
    return -log_likelihood(model, data)