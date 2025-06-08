Kernels
==========

Kernels are used to define trends in distribution parameters with regards to specific covariates.
They can be as complex as necessary but we provide by default a set of common kernels that can be
used directly or as a base for more complex ones.


.. currentmodule:: pykelihood.kernels

.. rubric:: Class

.. autosummary::
   :toctree: generated/

   ~Kernel
   ~constant

.. rubric:: Methods

.. autosummary::
   :toctree: generated/

   ~Kernel.with_covariate

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   ~linear
   ~polynomial
   ~exponential
   ~exponential_ratio
   ~trigonometric
   ~hawkes
   ~linear_regression
   ~exponential_linear_regression
   ~polynomial_regression
   ~categories_qualitative
   ~hawkes
