pykelihood
==========

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
- condition the likelihood with any form of penalty,
- profile parameters with a penalised likelihood,
- fit joint distributions and point processes with self exciting or time dependent intensity,
- more to come...


------------
Installation
------------

Using pip (*soon*)
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

>>> n.fit(data)
Normal(loc=1.0250822420920338, scale=1.9376400770300832)

As you can see, the values are slightly different from the moments in the data.
This is due to the fact that the ``fit`` method returns the Maximum Likelihood Estimator (MLE)
for the data, and is thus the result of an optimisation (using **scipy.optimize**).

We can also fix the value for some parameters if we know them:

>>> n.fit(data, loc=1)
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

>>> n.fit(data)
Normal(loc=-3.6462053656578005e-05, scale=0.5789668679237372)

Let's fit a ``Normal`` distribution with a trend in the loc parameter:

>>> from pykelihood import kernels
>>> n.fit(data, loc=kernels.linear(np.arange(365)))
Normal(loc=linear(a=-1.0000458359290572, b=0.005494714384381866), scale=0.0010055323717468906)

``kernels.linear(X)`` builds a linear model in the form :math:`a + bX` where :math:`a` and :math:`b` are parameters to
be optimised for, and :math:`X` is some covariate used to fit the data. If we assume the data were daily observations,
then we find all the values we expected: :math:`-1` was the value on the first day, :math:`0.05` was the daily increment
(:math:`2 / 365 = 0.05`), and there was a noise with std deviation :math:`0.001`.


Why do I have to create an instance to be able to fit my data?
**************************************************************

In the above example, we didn't specify any value or trend for the ``scale`` parameter. The reason it still worked is
that ``pykelihood`` assumed `scale` would have the same *form* as ``n``'s scale, which in this case is a simple float
parameter. Hence using an instance to fit the data avoids having to give a value for all parameters.

In some cases, it can become tedious to write everything out in one statement:

>>> from pykelihood.distributions import Beta
>>> X = np.arange(365)
>>> b = Beta(loc=kernels.linear(X), scale=kernels.linear(X), alpha=kernels.linear(X), beta=kernels.linear(X))

To avoid having so many parameters to optimise, you could decide to fix some parameters:

>>> b.fit(data, loc=0)
...
>>> b.fit(data, loc=1)
...

This syntax allows you to keep ``scale``, ``beta`` and ``alpha`` as linear trends while varying the value for the
``loc`` parameter.


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
    poetry install -D

This will create a virtual environment for the project and install the required dependencies. To activate the virtual
environment, be sure to run :code:`poetry shell` prior to executing any code.

Some parts of the code base use the `matplotlib <https://matplotlib.org/>`_ and
`hawkeslib <https://hawkeslib.readthedocs.io/en/latest/index.html>`_ package, but are for now not required to run most
of the code, including the tests.

Tests
-----

Tests are run using `pytest <https://docs.pytest.org/en/stable/>`_. To run all tests, navigate to the root folder or the
``tests`` folder and type :code:`pytest`.
