Fitting with Penalties
=======================

Another useful feature of ``pykelihood`` is the ability to customize the log-likelihood function with penalties,
conditioning methods, stability conditions, etc. Most statistics-related packages offer to fit data using the standard
opposite log-likelihood function, or in the best case, preselected models. To our knowledge, ``pykelihood`` is the only
Python package allowing to easily customize the log-likelihood function to fit data.

For example, say we want to penalize the target distribution parameters which L1-norm is too large: we would then
apply a Lasso penalty.

>>> data = np.random.normal(0, 1, 1000)
>>> def lassolike_score(distribution, data):
...     return -np.sum(distribution.logpdf(data)) + 5 * np.abs(distribution.loc())
...
>>> cond_fit = Normal().fit(data, score=lassolike_score)

We then compare a fit using the standard negative log-likelihood function to the use of the Lasso-penalized likelihood.

>>> std_fit = Normal().fit(data)
>>> std_fit.loc.value
-0.010891307380632494
>>> cond_fit.loc.value
-0.006210406541824357

The outcomes show that the penalty has been taken into account; the ``loc`` parameter of the distribution applying the penalty is smaller than with the standard opposite log-likelihood function.
