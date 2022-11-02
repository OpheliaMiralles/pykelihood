pykelihood
==========


.. image:: https://badge.fury.io/py/pykelihood.svg
    :target: https://pypi.org/project/pykelihood/

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

-----
About
-----

Pykelihood is a Python package for statistical analysis designed to give more flexibility to likelihood-based inference
than is possible with **scipy.stats**. Distributions are designed from an Object Oriented Programming (OOP) point of
view.

Main features include:

- use any **scipy.stats** distribution, or make your own,
- fit distributions of arbitrary complexity to your data,
- add trends of different forms in the parameters of any distribution,
- condition the log-likelihood with any form of penalty,
- profile parameters with a penalised log-likelihood,
- more to come...


------------
Installation
------------

Using pip
------------------

.. code::

    pip install pykelihood


From sources
------------

.. code::

    git clone https://www.github.com/OpheliaMiralles/pykelihood

or

.. code::

    gh repo clone OpheliaMiralles/pykelihood


-----
Usage
-----

Basics
------

The most basic thing you can use ``pykelihood`` for is creating and manipulating distributions as objects.

>>> from pykelihood.distributions import Normal
>>> n = Normal(1, 2)
>>> n
Normal(loc=1.0, scale=2.0)

``n`` is an *object* of type ``Normal``. It has 2 parameters, ``loc`` and ``scale``. They can be accessed like standard
Python attributes:

>>> n.loc
1.0

Using the ``Normal`` object, you can calculate standard values using the same semantics as **scipy.stats**:

>>> n.pdf([0, 1, 2])
array([0.17603266, 0.19947114, 0.17603266])
>>> n.cdf([0, 1, 2])
array([0.30853754, 0.5       , 0.69146246])

Or you can also generate random values according to this distribution:

>>> n.rvs(10)
array([ 3.31370986,  5.02699468, -0.3573229 ,  1.00460378, -3.26044871,
        1.86362711, -0.84192901,  0.81132182, -2.03266978,  1.48079944])


Fitting
-------

Let's generate a larger sample from our previous object:

>>> data = n.rvs(1000)
>>> data.mean()
1.025039359276458
>>> data.std()
1.9376460645596842

We can fit a ``Normal`` distribution to this data, which will return another ``Normal`` object:

>>> Normal.fit(data)
Normal(loc=1.0250822420920338, scale=1.9376400770300832)

As you can see, the values are slightly different from the moments in the data.
This is due to the fact that the ``fit`` method returns the Maximum Likelihood Estimator (MLE)
for the data, and is thus the result of an optimisation (using **scipy.optimize**).

We can also fix the value for some parameters if we know them:

>>> Normal.fit(data, loc=1)
Normal(loc=1.0, scale=1.9377929687500024)

Trend fitting
*************

One of the most powerful features of ``pykelihood`` is the ability to fit arbitrary distributions.
For instance, say our data has a linear trend with a very little gaussian noise we would like to capture:

>>> import numpy as np
>>> data = np.linspace(-1, 1, 365) + np.random.normal(0, 0.001, 365)
>>> data[:10]
array([-0.99802364, -0.99503679, -0.98900434, -0.98277981, -0.979487  ,
       -0.97393519, -0.96853445, -0.96149152, -0.95564004, -0.95054887])

If we try to fit this without a trend, the resulting distribution will miss out on most of the information:

>>> Normal.fit(data)
Normal(loc=-3.6462053656578005e-05, scale=0.5789668679237372)

Let's fit a ``Normal`` distribution with a trend in the loc parameter:

>>> from pykelihood import kernels
>>> Normal.fit(data, loc=kernels.linear(np.arange(365)))
Normal(loc=linear(a=-1.0000458359290572, b=0.005494714384381866), scale=0.0010055323717468906)

``kernels.linear(X)`` builds a linear model in the form *a + bX* where *a* and *b* are parameters to
be optimised for, and *X* is some covariate used to fit the data. If we assume the data were daily observations,
then we find all the values we expected: *-1* was the value on the first day, *0.05* was the daily increment
(*2 / 365 = 0.05*), and there was a noise with std deviation *0.001*.


Fitting with penalties
**********************

Another useful feature of ``pykelihood`` is the ability to customize the log-likelihood function with penalties, conditioning methods, stability conditions, etc. Most statistics-related packages offer to fit data using the standard opposite log-likelihood function, or in the best case, preselected models. To our knowledge, ``pykelihood`` is the only Python package allowing to easily customize the log-likelihood function to fit data.

>>> data = np.random.normal(0, 1, 1000)
>>> def lassolike_score(distribution, data):
...     return -np.sum(distribution.logpdf(data)) + 5 * np.abs(distribution.loc())
...
>>> std_fit = Normal.fit(data)
>>> cond_fit = Normal.fit(data, score=lassolike_score)
>>> std_fit.loc.value
-0.010891307380632494
>>> cond_fit.loc.value
-0.006210406541824357

Parameter profiling
*******************

Likelihood based inference relies on parameter estimation. This is why it's important to quantify the sensitivity of a
chosen model to each of those parameters. The ``stats_utils`` module in ``pykelihood`` includes the ``Profiler``
class that allows to link a model to a set of observations by providing goodness of fit metrics and profiles for all
parameters.

>>> from pykelihood.profiler import Profiler
>>> from pykelihood.distributions import GEV
>>> fitted_gev = GEV.fit(data, loc=kernels.linear(np.linspace(-1, 1, len(data))))
>>> ll = Profiler(fitted_gev, data, inference_confidence=0.99) # level of confidence for tests
>>> ll.AIC  # the standard fit is without trend
{'AIC MLE': -359.73533182968777, 'AIC Standard MLE Fit': 623.9896838880583}
>>> ll.profiles.keys()
dict_keys(['loc_a', 'loc_b', 'scale', 'shape'])
>>> ll.profiles["shape"].head(5)
      loc_a     loc_b     scale     shape   likelihood
0 -0.000122  1.000812  0.002495 -0.866884  1815.022132
1 -0.000196  1.000795  0.001964 -0.662803  1882.043541
2 -0.000283  1.000477  0.001469 -0.458721  1954.283256
3 -0.000439  1.000012  0.000987 -0.254640  2009.740282
4 -0.000555  1.000016  0.000948 -0.050558  1992.812843

Confidence intervals can be computed for specified metrics:

>>> def metric(gev): return gev.loc()
>>> ll.confidence_interval(metric)
[-4.160287666875364, 4.7039931595123825]


------------
Contributing
------------

`Poetry <http://python-poetry.org>`_ is used to manage ``pykelihood``'s dependencies and build system. To install
Poetry, you can refer to the `installation instructions <https://python-poetry.org/docs/#installation>`_, but it boils
down to running:

.. code::

    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python



To configure your environment to work on pykelihood, run:

.. code-block::

    git clone https://www.github.com/OpheliaMiralles/pykelihood  # or any other clone method
    cd pykelihood
    poetry install

This will create a virtual environment for the project and install the required dependencies. To activate the virtual
environment, be sure to run :code:`poetry shell` prior to executing any code.

We also use the `pre-commit <https://pre-commit.com>`_ library which adds git hooks to the repository. These must be installed with:

.. code::

   pre-commit install

Some parts of the code base use the `matplotlib <https://matplotlib.org/>`_ and
`hawkeslib <https://hawkeslib.readthedocs.io/en/latest/index.html>`_ package, but are for now not required to run most
of the code, including the tests.

Tests
-----

Tests are run using `pytest <https://docs.pytest.org/en/stable/>`_. To run all tests, navigate to the root folder or the
``tests`` folder and type :code:`pytest`.
