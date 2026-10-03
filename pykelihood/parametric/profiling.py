"""Profile likelihood tools for explicit-state fit results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import chi2

from pykelihood.likelihood import negative_log_likelihood
from pykelihood.parameters import Parameter
from pykelihood.parametric.fitting import (
    FitResult,
    NonFiniteInitialLikelihood,
    _fit_mle,
)
from pykelihood.state import ParameterLayout, State


@dataclass(frozen=True)
class ProfilePoint:
    """A profile score and nuisance refit at one fixed parameter value."""

    value: float
    fit_result: FitResult

    @property
    def log_likelihood(self) -> float:
        """Return the likelihood found by the nuisance refit."""
        return -float(self.fit_result.optimize_result.fun)

    @property
    def state(self) -> State:
        """Return the physical parameter state from this nuisance refit."""
        return self.fit_result.state


class Profiler:
    """Profile scalar parameters from an existing maximum-likelihood fit."""

    def __init__(self, fit_result: FitResult, *, confidence: float = 0.95) -> None:
        if not fit_result.optimize_result.success:
            raise ValueError("Profiling requires a successful initial fit.")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        self.fit_result = fit_result
        self.confidence = float(confidence)
        self._parameters = ParameterLayout.from_expr(fit_result.model).parameters
        self._cache: dict[tuple[Parameter, float], ProfilePoint] = {}
        self._max_log_likelihood = -negative_log_likelihood(
            fit_result.model, fit_result.data, state=fit_result.state
        )
        if not np.isfinite(self._max_log_likelihood):
            raise ValueError("Profiling requires a finite initial likelihood.")
        self._log_likelihood_threshold = (
            self._max_log_likelihood - float(chi2.ppf(confidence, df=1)) / 2.0
        )

    @property
    def max_log_likelihood(self) -> float:
        """Log likelihood at the fit result used to initialize the profiler."""
        return self._max_log_likelihood

    @property
    def log_likelihood_threshold(self) -> float:
        """Likelihood-ratio cutoff for the configured confidence level."""
        return self._log_likelihood_threshold

    def _validate_parameter(self, parameter: Parameter) -> None:
        if not isinstance(parameter, Parameter) or parameter not in self._parameters:
            raise ValueError("parameter must be a Parameter node in the fitted model.")
        if parameter.shape != ():
            raise ValueError("Only scalar parameters can be profiled.")
        if parameter in self.fit_result.fixed:
            raise ValueError(
                "A parameter fixed in the original fit cannot be profiled."
            )

    def _profile_one(self, parameter: Parameter, value: float) -> ProfilePoint:
        key = (parameter, float(value))
        if key in self._cache:
            return self._cache[key]

        fixed = dict(self.fit_result.fixed)
        fixed[parameter] = np.asarray(value, dtype=np.float64)
        nearby = [
            point for (node, _), point in self._cache.items() if node is parameter
        ]
        starting_state = (
            min(nearby, key=lambda point: abs(point.value - value)).state
            if nearby
            else self.fit_result.state
        )
        profiled_fit = _fit_mle(
            self.fit_result.model,
            self.fit_result.data,
            state=starting_state,
            fixed=fixed,
        )
        if not profiled_fit.optimize_result.success or not np.isfinite(
            profiled_fit.optimize_result.fun
        ):
            raise RuntimeError(f"Nuisance fit failed at profile value {value}.")
        point = ProfilePoint(float(value), profiled_fit)
        self._cache[key] = point
        return point

    def profile(
        self, parameter: Parameter, values: npt.ArrayLike
    ) -> tuple[ProfilePoint, ...]:
        """Fix each candidate and refit nuisance parameters from nearby fits."""
        self._validate_parameter(parameter)
        candidates = np.asarray(values, dtype=np.float64)
        if candidates.ndim == 0:
            candidates = candidates.reshape(1)
        if candidates.ndim != 1:
            raise ValueError("Profile candidate values must be one-dimensional.")
        if not np.all(np.isfinite(candidates)):
            raise ValueError("Profile candidate values must be finite.")
        return tuple(self._profile_one(parameter, float(value)) for value in candidates)

    def confidence_interval(
        self,
        parameter: Parameter,
        *,
        step: float | None = None,
        precision: float = 1e-5,
        max_expansions: int = 100,
    ) -> tuple[float, float]:
        """Find the likelihood-ratio interval by bracketing and root solving.

        The default first step is positive even when the fitted value is zero;
        each subsequent step doubles until the profile falls below the cutoff.
        """
        self._validate_parameter(parameter)
        if not np.isfinite(precision) or precision <= 0.0:
            raise ValueError("precision must be finite and positive.")
        if max_expansions < 1:
            raise ValueError("max_expansions must be positive.")

        center = float(np.asarray(self.fit_result.state[parameter]))
        initial_step = max(0.1 * abs(center), 0.1) if step is None else float(step)
        if not np.isfinite(initial_step) or initial_step <= 0.0:
            raise ValueError("step must be finite and positive.")
        center_point = self._profile_one(parameter, center)
        if center_point.log_likelihood < self._log_likelihood_threshold:
            raise RuntimeError("The fitted optimum is below its own profile cutoff.")

        def score(value: float) -> float | None:
            if not np.isfinite(value):
                return None
            try:
                return self._profile_one(parameter, value).log_likelihood
            except NonFiniteInitialLikelihood:
                return None

        def find_bracket(direction: float) -> tuple[float, float]:
            inside = center
            distance = initial_step
            for _ in range(max_expansions):
                outside = center + direction * distance
                outside_score = score(outside)
                if outside_score is None:
                    valid = inside
                    invalid = outside
                    for _ in range(64):
                        midpoint = (valid + invalid) / 2.0
                        if midpoint == valid or midpoint == invalid:
                            break
                        midpoint_score = score(midpoint)
                        if midpoint_score is not None:
                            if midpoint_score < self._log_likelihood_threshold:
                                return (
                                    (midpoint, inside)
                                    if direction < 0
                                    else (inside, midpoint)
                                )
                            valid = midpoint
                        else:
                            invalid = midpoint
                    side = "lower" if direction < 0 else "upper"
                    raise RuntimeError(
                        f"Unable to bracket the {side} confidence limit before an "
                        "unscorable refit. A failed nuisance starting state need "
                        "not indicate a parameter domain boundary."
                    )
                if outside_score < self._log_likelihood_threshold:
                    return (outside, inside) if direction < 0 else (inside, outside)
                inside = outside
                distance *= 2.0
            side = "lower" if direction < 0 else "upper"
            raise RuntimeError(f"Unable to bracket the {side} confidence limit.")

        lower_bracket = find_bracket(-1.0)
        upper_bracket = find_bracket(1.0)

        def difference(value: float) -> float:
            return (
                self._profile_one(parameter, value).log_likelihood
                - self._log_likelihood_threshold
            )

        lower = cast(float, brentq(difference, *lower_bracket, xtol=precision))
        upper = cast(float, brentq(difference, *upper_bracket, xtol=precision))
        return float(lower), float(upper)
