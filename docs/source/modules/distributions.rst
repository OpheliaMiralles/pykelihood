Distributions
=============

Most distributions are wrapped around SciPy distribution objects.
It is structured in a hierarchical way, where the parameters of a distribution can themselves have additional parameters.
The likelihood of the parameterized distribution for a given dataset is optimized using SciPy's ``minimize`` function.
Pykelihood distributions share methods and arguments with SciPy distributions, though occasionally,
some parameters have been adjusted (e.g., the GEV distribution) to align with statistical community standards.

.. currentmodule:: pykelihood.distributions

.. rubric:: Class

.. autosummary::
   :toctree: generated/

   ~Distribution

.. rubric:: Methods

.. autosummary::
   :toctree: generated/

   ~Distribution.cdf
   ~Distribution.fit
   ~Distribution.fit_instance
   ~Distribution.inverse_cdf
   ~Distribution.isf
   ~Distribution.logcdf
   ~Distribution.logpdf
   ~Distribution.logsf
   ~Distribution.param_dict_to_vec
   ~Distribution.param_mapping
   ~Distribution.pdf
   ~Distribution.ppf
   ~Distribution.rvs
   ~Distribution.sf
   ~Distribution.with_params


.. rubric:: Attributes

.. autosummary::
   :toctree: generated/

   ~Distribution.flattened_param_dict
   ~Distribution.flattened_params
   ~Distribution.optimisation_param_dict
   ~Distribution.optimisation_params
   ~Distribution.param_dict
   ~Distribution.params
   ~Distribution.params_names
