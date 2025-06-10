Parameters
==========

The parameters module provides a framework for defining and managing parameters
used in statistical models and distributions. It allows for the creation of parameter objects
that can be optimized, ensuring that they are properly initialized.
Parameters can take any form, including constants or kernels.

.. currentmodule:: pykelihood.parameters

.. rubric:: Class

.. autosummary::
   :toctree: generated/

   ~Parameter
   ~ConstantParameter

.. rubric:: Methods

.. autosummary::
   :toctree: generated/

   ~Parameter.with_params


.. rubric:: Attributes

.. autosummary::
   :toctree: generated/

   ~Parameter.params
   ~Parameter.optimisation_params
   ~Parameter.value

.. rubric:: Functions
.. autosummary::
   :toctree: generated/

   ~ensure_parametrized
