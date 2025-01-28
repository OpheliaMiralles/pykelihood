Trend Fitting
===============

One of the most powerful features of ``pykelihood`` is the ability to fit arbitrary distributions.
For instance, say our data has a linear trend with a very little gaussian noise we would like to capture.

>>> import numpy as np
>>> data = np.linspace(-1, 1, 365) + np.random.normal(0, 0.001, 365)
>>> data[:10]
array([-0.99802364, -0.99503679, -0.98900434, -0.98277981, -0.979487  ,
       -0.97393519, -0.96853445, -0.96149152, -0.95564004, -0.95054887])

If we try to fit this without a trend, the resulting distribution will miss out on most of the information.

>>> Normal.fit(data)
Normal(loc=-3.6462053656578005e-05, scale=0.5789668679237372)

Fitting a ``Normal`` distribution with a trend in the ``loc`` parameter can be done using the following piece of code:

>>> from pykelihood import kernels
>>> Normal.fit(data, loc=kernels.linear(np.arange(365)))
Normal(loc=linear(a=-1.0000458359290572, b=0.005494714384381866), scale=0.0010055323717468906)

The ``kernels`` module is flexible and can be adapted by users to support any kind of trend.
``kernels.linear(X)`` builds a linear model in the form *a + bX* where *a* and *b* are parameters to
be optimised for, and *X* is some covariate used to fit the data. If we assume the data were daily observations,
then we find all the values we expected: *-1* was the value on the first day, *0.05* was the daily increment
(*2 / 365 = 0.05*), and there was a noise with std deviation *0.001*.
