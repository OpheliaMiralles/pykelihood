Effects
=======

The :mod:`pykelihood.effects` module provides the building blocks for covariate-dependent
models. Effects can be composed using arithmetic operators and used directly as
distribution parameters.

.. code-block:: python

    import numpy as np
    from pykelihood.effects import linear

    x = np.linspace(0.0, 1.0, 5)
    trend = linear(slope=2.0).with_covariate(x)

    trend.eval({})

.. currentmodule:: pykelihood.effects

.. rubric:: Classes

.. autosummary::
   :toctree: generated/

   ~Effect
   ~FunctionEffect
   ~BoundEffect

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   ~build_effect
   ~define_effect
   ~constant
   ~linear
   ~polynomial
   ~categorical
   ~gaussian
   ~exp
