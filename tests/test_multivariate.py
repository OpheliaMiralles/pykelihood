from copulae.archimedean.gumbel import GumbelCopula
from pytest import approx

from pykelihood.distributions import Exponential, Normal
from pykelihood.multivariate import Copula, Multivariate
from pykelihood.parameters import Parameter


def test_multivariate():
    copula = Copula(GumbelCopula)
    m1 = Normal()
    m2 = Exponential()
    mv = Multivariate(m1, m2, copula=copula)
    mv_with_theta = mv.with_params([0, 1, 0, 1, 3])
    assert mv_with_theta.copula.params[0].value == 3
    assert mv_with_theta.rvs(10).shape == (10, 2)


def test_multivariate_fit():
    copula = Copula(GumbelCopula, theta=Parameter(3))
    m1 = Normal()
    m2 = Exponential()
    mv = Multivariate(m1, m2, copula=copula)
    data = mv.rvs(10000)

    new = mv.fit(data)
    assert isinstance(new.copula, Copula)
    assert new.copula.theta.value == approx(3, rel=0.05)
