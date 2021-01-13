import numpy as np
import pytest
from scipy.stats import genextreme

from pykelihood import distributions, kernels
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

