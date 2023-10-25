# Changelog

## 0.4.1 - 2023-10-25

### New features

* The `Distribution.fit` method accepts a `scipy_args` dictionary which is
  passed to `scipy`'s `minimize` function.
* The confidence interval computed by the profiler now uses root finding to
  find the bounds where the likelihood ratio test starts failing. This means
  confidence intervals can only be computed for the distribution's parameters.
* Upper bounds on dependencies were removed, improving compatibility with
  recent versions.

## 0.4.0 - 2023-10-14

### Breaking changes

* Renamed `stats_utils` module to `profiler`
* Data must now be provided to kernels on creation, unbound kernels are
  no longer allowed
* `Parameter`s are no longer subclasses of `float`, use `.value` to get
  their stored value
* `ConditioningMethod`s were removed, their uses can be replaced with
  _score functions_
* The `biv` parameter to the `Profiler` was removed, confidence
  intervals are univariate only

### Removed

Many distributions and utilities which were created with a specific use
case in mind and aren't generally useful have been removed:

* `MixtureExponentialModel`,
* `ExtendedGPD`,
* `PointProcess`,
* `CompositionDistribution`,
* `DetrendedFluctuationAnalysis`,
* `pettitt_test`,
* `threshold_selection_GoF` and `threshold_selection_gpd_NorthorpColeman`,
* extreme values visualisation routines,
* process samplers (Poisson and Hawkes).

### New features

* Metrics: `{pp,qq}_l{1,2}_distance`, `likelihood`, `expo_ratio`
* Log-normal distribution
* Plotting functions now accept an `ax` argument to use instead of the
  global `plt` figure
* Constant kernel (most useful for testing)
* `Kernel`s have a `with_covariate` method that returns a new kernel
  with the provided data as covariate, but all parameters are kept the
  same
* The `random_state` parameter to the `Distribution.rvs` method is now
  explicit and no longer hidden in the `**kwargs`

### Bug fixes

* Fixed `fit_instance` for nested kernels with fixed values
* Fixed the `TruncatedDistribution` which forgot its bounds after fitting
* A parameter which shows up in several places in a distribution will
  keep the same value when fitting instead of returning independent
  parameters

### Other

* Add section to README on fitting other score functions than the likelihood
* Add changelog with all version changes up to this one

## 0.3.2 - 2021-03-26

Various bug fixes due to new names in version 0.3.0

## 0.3.1 - 2021-03-26

### New features

* New trigonometric kernel

## 0.3.0 - 2021-03-23

This release aims at making fitting even more generic by replacing
penalized likelihoods with score functions

### Breaking changes

* Remove `log_likelihood` and `opposite_log_likelihood` methods from
 `Distribution`s
* Remove `penalty` argument from `fit`
* `ConditioningMethod`s take `distribution` as the first argument and
 `data` as the second
* Rename `Likelihood` class to `Profiler`
* Rename `mle` attribute to `optimum` in `Profiler` class

### New features

* `fit*` methods minimize a `score` function which takes a distribution
  and data as arguments. By default, they maximize the log likelihood.
* New `metrics` module which contains scoring functions
* The `Profiler` has a `score_function` argument
* Add `threshold_selection` based on a multiple threshold penultimate model
* New QQ-plot visualization methods

### Bug fixes
* `Distribution.rvs` now passes all parameters to `scipy`'s `rvs` methods.
  In particular, `random_state` is now propagated.
* `TruncatedDistribution` now ignores data outside its range

### Other

* Improve overall typing of the library

## 0.2.1 - 2021-02-21

### New features

* Add `TruncatedDistribution`

### Bug fixes

* Avoid replacing non-constant parameters with `ConstantParameter`s
  inside `ParametrizedFunction`s
* Fix nested kernel fitting

## 0.2.0 - 2021-02-09

This release aims at making the fitting feature more generic by allowing
other forms of likelihood conditioning and other metrics than the return
level in the confidence interval computation.

### Breaking changes

* Remove `simulations` and `stopping_times` modules
* Removed `Likelihood.return_level`
* Move `visualisation_utils` to their own subpackage
* Rename `conditioning_method` argument to `penalty`
* Rename `profile_likelihood` method to `fit_instance`

### New features

* New method `Likelihood.confidence_interval` with the `metric` argument
  to control which value to estimate

### Other

* Add section to README.rst about parameter profiling
* Add pre-commit hooks
* Fix README.rst directives


## 0.1.0 - 2021-02-04

Initial release
