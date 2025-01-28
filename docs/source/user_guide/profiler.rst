Parameter profiling
====================

Likelihood-based inference requires parameter estimation, so it is important to quantify the sensitivity of a chosen model
to each of those parameters. The ``profiler`` module includes the ``Profiler`` class that allows the linking of a model
to a set of observations by providing some goodness of fit metrics and "profiles" for all parameters. Profiles are
provided under the form of a dictionary of ``pandas.DataFrame`` objects. Each key is a parameter to profile, i.e. to fix
and vary while the other distribution parameters are optimized, and each associated data frame contains the values of all
of the distribution parameters as well as this of the ``score`` function (usually the opposite log-likelihood) throughout
the partial optimization.

If the distribution includes a trend in one of the parameters, the parameters of the trend will be profiled. If some
parameters were fixed in the distribution provided to the ``Profiler``, the associated profiles are not computed.
Computing the profile likelihood can be done as follows.

>>> from pykelihood.profiler import Profiler
>>> from pykelihood.distributions import GEV
>>> fitted_gev = GEV.fit(data, loc=kernels.linear(np.linspace(-1, 1, len(data))))
>>> profiler = Profiler(fitted_gev, data, inference_confidence=0.99) # level of confidence for tests
>>> profiler.AIC  # the standard fit is without trend
{'AIC MLE': -359.73533182968777, 'AIC Standard MLE Fit': 623.9896838880583}
>>> profiler.profiles.keys()
dict_keys(['loc_a', 'loc_b', 'scale', 'shape'])
>>> profiler.profiles["shape"].head(5)
      loc_a     loc_b     scale     shape   score
0 -0.000122  1.000812  0.002495 -0.866884  1815.022132
1 -0.000196  1.000795  0.001964 -0.662803  1882.043541
2 -0.000283  1.000477  0.001469 -0.458721  1954.283256
3 -0.000439  1.000012  0.000987 -0.254640  2009.740282
4 -0.000555  1.000016  0.000948 -0.050558  1992.812843

A binary search algorithm implemented to compute the parameter confidence intervals allows for very efficient exploration
of the parameter space. It can be provided with a ``precision`` argument, defaulted to *10^{-5}*.
For example, if the parameter of interest is the location of the GEV distribution, the profile likelihood-based associated
confidence interval is computed using the following syntax:

>>> profiler.confidence_interval("loc", precision=1e-3)

from which the output would be an array containing the lower and upper bound for the corresponding confidence interval (using the level defined as a parameter of the ``Profiler`` object).

>>> [-4.160287666875364, 4.7039931595123825]
