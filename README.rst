pykelihood
===========


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
view. In particular, this package allows the fitting of complex distributions to a dataset, add trends of different forms
in the parameters of the target distribution, condition the log-likelihood with any form of penalty, and profile
parameters of the model based on the chosen likelihood's sensitivity.

Main features include:

- use any **scipy.stats** distribution, or make your own,
- fit distributions of arbitrary complexity to your data,
- add trends of different forms in the parameters of any distribution,
- condition the log-likelihood with any form of penalty,
- profile parameters with a penalised log-likelihood,
- more to come...

The toy examples presented in the documentation may seem simple, but the package is highly practical in many cases.
It allows for fitting trends, conditioning log-likelihoods with custom penalty functions, and reparameterizing distributions
like the GEV in terms of return levels (which is essential in the field of extreme event attribution, for instance).
It also simplifies tasks such as building profile likelihood-based confidence intervals, making it a flexible and efficient
tool for statistical modeling.

-------------
Documentation
-------------
For detailed documentation, please visit the `official documentation <https://pykelihood.readthedocs.io>`_.


Installation
------------

From PyPI
------------------

.. code-block:: console

    pip install pykelihood

------------
Contributing
------------

`UV <https://docs.astral.sh/uv/>`_ is used to manage ``pykelihood``'s dependencies and build system. To install
UV, you can refer to the `installation instructions <https://docs.astral.sh/uv/getting-started/installation/>`_.

To configure your environment to work on pykelihood, run:

.. code-block::

    git clone https://www.github.com/OpheliaMiralles/pykelihood
    cd pykelihood
    uv sync

This will create a virtual environment for the project and install the required dependencies.

We also use the `pre-commit <https://pre-commit.com>`_ tool which manages git hooks for the repository. It can be installed
with:

.. code-block::

   uv sync --group tools

Then, to install the hooks, run:

.. code-block::

   pre-commit install

Tests
-----

Tests are run using `pytest <https://docs.pytest.org/en/stable/>`_. To run all tests, navigate to the root folder and type :code:`pytest`.
