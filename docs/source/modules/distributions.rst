Distributions
=============

Most distributions are wrapped around SciPy distribution objects.
It is structured in a hierarchical way, where the parameters of a distribution can themselves have additional parameters.
The likelihood of the parameterized distribution for a given dataset is optimized using SciPy's ``minimize`` function.
Pykelihood distributions share methods and arguments with SciPy distributions, though occasionally, some parameters have been adjusted (e.g., the GEV distribution) to align with statistical community standards.

.. automodule:: pykelihood.distributions
   :members:
   :noindex:
