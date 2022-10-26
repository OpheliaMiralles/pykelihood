from itertools import chain
from typing import Sequence, Type

import numpy as np
import scipy.optimize
from copulae.archimedean.abstract import (
    AbstractArchimedeanCopula as _AbstractArchimedeanCopula,
)
from copulae.marginal.marginal import MarginalCopula as _MarginalCopula

from pykelihood.distributions import Distribution
from pykelihood.generic_types import Obs
from pykelihood.metrics import opposite_log_likelihood
from pykelihood.parameters import Parameter, ensure_parametrized


class Copula:
    def __init__(
        self,
        base_copula_class: Type[_AbstractArchimedeanCopula],
        dim: int = 2,
        theta: float = None,
    ):
        self._base = base_copula_class
        self.dim = dim
        self.theta = (
            ensure_parametrized(theta, constant=True)
            if theta is not None
            else Parameter()
        )

    def _base_copulae_instance(self) -> _AbstractArchimedeanCopula:
        return self._base(self.theta.value, self.dim)

    def cdf(self, u: Obs):
        return self._base_copulae_instance().cdf(u)

    def pdf(self, u: Obs):
        return self._base_copulae_instance().pdf(u)

    def logpdf(self, u: Obs):
        return self._base_copulae_instance().pdf(u, log=True)

    def rvs(self, size: int = None) -> np.ndarray:
        return self._base_copulae_instance().random(size)

    @property
    def params(self):
        return (self.theta,)

    def with_params(self, params) -> "Copula":
        return Copula(self._base, self.dim, self.theta.with_params(params))

    def fit_instance(
        self,
        data,
        score=opposite_log_likelihood,
        x0: Sequence[float] = None,
        **fixed_values,
    ) -> "Copula":
        base = self._base_copulae_instance()
        base.fit(data, x0)
        return Copula(self._base, self.dim, theta=base.params)


class _MarginalCopulaAdapter(_MarginalCopula):
    def __init__(
        self, copula: _AbstractArchimedeanCopula, marginals: Sequence[Distribution]
    ):
        # Don't call super.__init__, instead set the attributes directly
        self._copula = copula
        self._marginals = marginals
        self._dim = len(marginals)


class Multivariate:
    def __init__(self, *marginals: Distribution, copula: Copula):
        if len(marginals) != copula.dim:
            raise ValueError(
                f"Got {len(marginals)} marginals but copula has {copula.dim} dimensions"
            )
        self.marginals = marginals
        self.copula = copula

    def _copulae_adapter(self) -> _MarginalCopulaAdapter:
        return _MarginalCopulaAdapter(
            self.copula._base_copulae_instance(), self.marginals
        )

    def cdf(self, x: Obs):
        return self._copulae_adapter().cdf(x)

    def pdf(self, x: Obs):
        return self._copulae_adapter().pdf(x)

    def logpdf(self, x: Obs):
        return self._copulae_adapter().pdf(x, log=True)

    def rvs(self, size: int) -> np.ndarray:
        return self._copulae_adapter().random(size)

    @property
    def params(self):
        return (
            tuple(chain.from_iterable(m.flattened_params for m in self.marginals))
            + self.copula.params
        )

    def with_params(self, params) -> "Multivariate":
        params = iter(params)
        marginals = [m.with_params(params) for m in self.marginals]
        copula = self.copula.with_params(params)
        return Multivariate(*marginals, copula=copula)

    def fit(
        self,
        data,
        score=opposite_log_likelihood,
    ):
        x0 = [p.value for p in self.params]

        def to_minimize(x):
            obj = self.with_params(x)
            try:
                return score(obj, data)
            except Exception:
                return np.inf

        res = scipy.optimize.minimize(to_minimize, x0)

        opt_params = res.x
        return self.with_params(opt_params)
