import pytest
from pykelihood import distributions, kernels
import numpy as np

def test_fit_R_distribution_with_trend():
    """
    Test designed to check if the operations in R are correctly performed when
    we input a trend, ie pointwise cdf, pdf, ...
    :return:
    """
    distribution = distributions.RGEV(0., 1., 0.2)
    data = distribution.rvs(1000)
    covariate = np.arange(1000)
    fit_with_linear_trend = distribution.fit(data, loc=kernels.linear(covariate))
    for i in range(len(fit_with_linear_trend.loc())):
        distro = type(distribution)(fit_with_linear_trend.loc()[i],
                              fit_with_linear_trend.scale(),
                              fit_with_linear_trend.shape())
        value = np.array(distro.cdf(data[i]))[0]
        assert np.array(fit_with_linear_trend.cdf(data))[i] == pytest.approx(value)
        value = np.array(distro.pdf(data[i]))[0]
        assert np.array(fit_with_linear_trend.pdf(data))[i] == pytest.approx(value)
        value = np.array(distro.isf(0.99))[0]
        assert np.array(fit_with_linear_trend.isf(0.99))[i] == pytest.approx(value)
