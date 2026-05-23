Kernels
=======

.. warning::

   For new code, prefer using :mod:`pykelihood.effects` directly. The kernel functions here
   are thin wrappers maintained for backward compatibility.

Kernels are used to define trends in distribution parameters with regards to specific covariates.
They can be as complex as necessary but we provide by default a set of common kernels that can be
used directly or as a base for more complex ones.

.. code-block:: python

    import numpy as np
    from pykelihood import kernels
    from pykelihood.distributions import Normal

    x = np.linspace(0.0, 1.0, 100)
    loc = kernels.linear(x, a=1.0, b=2.0)
    scale = kernels.constant(0.5)

    model = Normal(loc=loc, scale=scale)
    sample = model.rvs(10)

The same module also covers polynomial, trigonometric, regression-style, and
categorical helpers.

.. currentmodule:: pykelihood.kernels

.. rubric:: Class

.. autosummary::
   :toctree: generated/

   ~Kernel

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   ~constant
   ~linear
   ~polynomial
   ~exponential
   ~exponential_ratio
   ~gaussian
   ~trigonometric
   ~linear_regression
   ~exponential_linear_regression
   ~polynomial_regression
   ~categories_qualitative
   ~hawkes
