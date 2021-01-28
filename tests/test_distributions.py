import numpy as np
import pytest
from scipy.stats import genextreme

from pykelihood import kernels
from pykelihood.distributions import GEV, Normal

REL_PREC = 1e-7
ABS_PREC = 0.1


def approx(x):
    return pytest.approx(x, rel=REL_PREC, abs=ABS_PREC)


class TestGEV:
    def test_fit(self, datasets):
        for ds in datasets:
            c, loc, scale = genextreme.fit(ds)
            fit = GEV.fit(ds)
            assert fit.loc == approx(loc)
            assert fit.scale == approx(scale)
            assert fit.shape == approx(-c)

    def test_fixed_values(self):
        data = np.random.standard_normal(1000)
        raw = Normal.fit(data)
        assert raw.loc == approx(0.)
        assert raw.scale == approx(1.)
        fixed = Normal.fit(data, loc=1.)
        assert fixed.loc == 1.


def test_cache():
    n = Normal(0, 1)
    np.testing.assert_array_almost_equal(n.pdf([-1, 0, 1]), [0.24197072, 0.39894228, 0.24197072])
    np.testing.assert_array_almost_equal(n.pdf([1, 2, 3]), [0.24197072, 0.05399097, 0.00443185])
    np.testing.assert_array_almost_equal(n.cdf([-1, 0, 1]), [0.15865525, 0.5, 0.84134475])
    np.testing.assert_array_almost_equal(n.pdf([-1, 0, 1]), [0.24197072, 0.39894228, 0.24197072])


def test_basic_with_params():
    n = Normal()
    m = n.with_params([2, 3])
    assert m.loc == 2
    assert m.scale == 3


def test_named_with_params():
    n = Normal()
    m = n.with_params(loc=2, scale=3)
    assert m.loc == 2
    assert m.scale == 3


def test_named_with_params_multi_level():
    n = Normal(loc=kernels.linear(), scale=1)
    m = n.with_params(loc_a=2, scale=3)
    assert m.loc.a == 2
    assert m.scale == 3


def test_named_with_params_partial_assignment():
    n = Normal()
    m = n.with_params(scale=3)
    assert m.loc == 0
    assert m.scale == 3
