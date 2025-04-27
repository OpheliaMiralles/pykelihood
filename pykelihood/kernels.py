import re
from collections.abc import Collection, Sequence
from functools import wraps
from itertools import count
from typing import Union

import numpy as np
import pandas as pd

from pykelihood.parameters import Parameter, ParametrizedFunction, ensure_parametrized


class Kernel(ParametrizedFunction):
    """
    Represents a kernel function of one covariate with parameters.

    Parameters
    ----------
    f : callable
        The kernel function to be wrapped.
    x : array-like, optional
        Covariates on which the kernel will operate.
    fname : str, optional
        Name of the kernel function.
    params : dict
        Parameters of the kernel function.
    """

    def __init__(self, f, x=None, fname=None, **params):
        """Function of one covariate with parameters."""
        fname = fname or f.__qualname__
        super().__init__(f, fname=fname, **params)
        self._validate(x)
        self.x = x

    def _validate(self, x):
        """
        Validates the covariate input.

        Parameters
        ----------
        x : array-like
            Covariates to validate.

        Raises
        ------
        ValueError
            If `x` is not iterable.
        """
        try:
            iter(x)
        except TypeError:
            raise ValueError(
                f"Incorrect covariate for {self.fname}: {self.x}"
            ) from None

    def __call__(self, x=None):
        """
        Evaluate the kernel function on the given covariate.

        Parameters
        ----------
        x : array-like, optional
            Covariate values. If not provided, uses the instance's `x`.

        Returns
        -------
        float
            Result of the kernel function evaluation.
        """
        if x is None:
            x = self.x
        self._validate(x)
        param_values = {p_name: p() for p_name, p in self.param_dict.items()}
        return self.f(x, **param_values)

    def _build_instance(self, **new_params):
        return type(self)(self.f, self.x, fname=self.fname, **new_params)

    def with_covariate(self, covariate):
        """
        Create a new instance of the kernel with the given covariate.

        Parameters
        ----------
        covariate : array-like
            New covariate values.

        Returns
        -------
        Kernel
            New kernel instance with updated covariate.
        """
        return type(self)(self.f, covariate, fname=self.fname, **self.param_dict)


class constant(Kernel):
    """
    A kernel representing a constant value.

    Parameters
    ----------
    value : float, optional
        Constant value for the kernel. Default is 0.0.
    """

    def __init__(self, value=0.0):
        super().__init__(self._call, fname="constant", value=value)

    def _build_instance(self, value):
        return constant(value)

    def _validate(self, x):
        """
        Validates the input for the constant kernel.

        Parameters
        ----------
        x : any
            Input value to validate.

        Raises
        ------
        ValueError
            If `x` is not None.
        """
        if x is not None:
            raise ValueError(f"Unexpected data for constant kernel: {x}")

    def _call(self, _x, value):
        """
        Compute the constant kernel value.

        Parameters
        ----------
        _x : any
            Ignored input.
        value : float
            The constant value.

        Returns
        -------
        float
            The constant value.
        """
        return value


def kernel(**param_defaults):
    """
    Decorator for creating a kernel function with parameters.

    Parameters
    ----------
    param_defaults : dict
        Default values for the kernel parameters.

    Returns
    -------
    callable
        A decorated kernel function.
    """

    def wrapper(f):
        @wraps(f)
        def wrapped(x, **param_values) -> Kernel:
            final_params = {}
            for p_name, default_value in param_defaults.items():
                override = param_values.get(p_name)
                if override is not None:
                    final_params[p_name] = ensure_parametrized(override, constant=True)
                else:
                    final_params[p_name] = ensure_parametrized(default_value)
            return Kernel(f, x, fname=f.__name__, **final_params)

        return wrapped

    return wrapper


"""
Simple kernels with one covariate
"""


@kernel(a=0.0, b=0.0)
def linear(X, a, b):
    r"""
    Linear kernel function.

    .. math::

        y = a + b \cdot X

    Parameters
    ----------
    X : array-like
        Input data.
    a : float
        Intercept of the linear function.
    b : float
        Slope of the linear function.

    Returns
    -------
    array-like
        Output of the linear kernel.
    """
    return a + b * X


@kernel(a=0.0, b=0.0, c=0.0)
def polynomial(X, a, b, c):
    r"""
    Polynomial kernel function.

    .. math::

        y = a + b \cdot X + c \cdot X^2

    Parameters
    ----------
    X : array-like
        Input data.
    a : float
        Constant term.
    b : float
        Coefficient for the linear term.
    c : float
        Coefficient for the quadratic term.

    Returns
    -------
    array-like
        Output of the polynomial kernel.
    """
    return a + b * X + c * X**2


@kernel(a=0.0, b=0.0)
def exponential(X, a, b):
    r"""
    Exponential kernel function.

    .. math::

        y = \exp(a + b \cdot X)

    Parameters
    ----------
    X : array-like
        Input data.
    a : float
        Coefficient for the constant term inside the exponential.
    b : float
        Coefficient for the linear term inside the exponential.

    Returns
    -------
    array-like
        Exponential of the linear function.
    """
    inner = b * X
    inner = a + inner
    return np.exp(inner)


@kernel(a=0.0, b=1.0, c=1.0)
def exponential_ratio(X, a, b, c):
    r"""
    Exponential ratio kernel function.

    .. math::

        y = c \cdot \exp\left(\frac{a \cdot X}{b}\right)

    Parameters
    ----------
    X : array-like
        Input data.
    a : float
        Numerator coefficient inside the exponential.
    b : float
        Denominator coefficient inside the exponential.
    c : float
        Scaling factor.

    Returns
    -------
    array-like
        Exponential ratio kernel output.
    """
    inner = a * X
    inner = inner / b
    return c * np.exp(inner)


@kernel(mu=0.0, sigma=1.0, scaling=0.0)
def gaussian(X, mu, sigma, scaling):
    r"""
    Gaussian kernel function.

    .. math::

        y = \text{scaling} \cdot \frac{1}{\sigma \sqrt{2\pi}} \cdot
        \exp\left(-\frac{(X - \mu)^2}{2 \sigma^2}\right)

    Parameters
    ----------
    X : array-like
        Input data.
    mu : float
        Mean of the Gaussian function.
    sigma : float
        Standard deviation of the Gaussian function.
    scaling : float
        Scaling factor for the output.

    Returns
    -------
    array-like
        Gaussian kernel output.
    """
    mult = scaling * 1 / (sigma * np.sqrt(2 * np.pi))
    expo = np.exp(-((X - mu) ** 2) / sigma**2)
    return mult * expo


@kernel(a=0.0, b=0.0, c=0.0)
def trigonometric(X, a, b, c):
    r"""
    Trigonometric kernel function.

    .. math::

        y = a + b \cdot \cos(2\pi X) + c \cdot \sin(2\pi X)

    Parameters
    ----------
    X : array-like
        Rescaled input vector per period of interest.
    a : float
        Constant term.
    b : float
        Coefficient for the cosine term.
    c : float
        Coefficient for the sine term.

    Returns
    -------
    array-like
        Trigonometric kernel output.
    """
    return a + b * np.cos(2 * np.pi * X) + c * np.sin(2 * np.pi * X)


@kernel(mu=0.0, alpha=0.0, theta=1.0)
def hawkes_with_exp_kernel(X, mu, alpha, theta):
    r"""
    Hawkes process with exponential kernel.

    .. math::

        \lambda(t) = \mu + \alpha \cdot \sum_{t_i < t} \exp(-\theta (t - t_i))

    Parameters
    ----------
    X : array-like
        Times of occurrence of events.
    mu : float
        Background constant intensity.
    alpha : float
        Infectivity of events.
    theta : float
        Decay term describing the decrease in intensity over time.

    Returns
    -------
    array-like
        Intensity function values at each time point.
    """
    return mu + alpha * theta * np.array(
        [np.sum(np.exp(-theta * (X[i] - X[:i]))) for i in range(len(X))]
    )


def hawkes_exp_with_event_times(t, tau, mu, alpha, theta):
    r"""
    Computes the Hawkes process exponential kernel with given event times.

    Parameters
    ----------
    t : float
        The current time at which the kernel is evaluated.
    tau : np.ndarray
        An array of event times.
    mu : float
        The base intensity of the Hawkes process.
    alpha : float
        The self-excitation parameter, which controls the contribution of past events to the intensity.
    theta : float
        The decay rate of the exponential kernel.

    Returns
    -------
    float
        The intensity value of the Hawkes process at time `t`.

    Notes
    -----
    The intensity function is defined as:

    .. math::
        \lambda(t) = \mu + \alpha \theta \sum_{\tau_i < t} \exp(-\theta (t - \tau_i))

    where `tau` is the sequence of event times up to `t`.
    """
    return mu + alpha * theta * np.sum(
        (np.exp(-theta * (t - tau[i]))) for i in range(len(tau)) if tau[i] < t
    )


"""
Sophisticated kernels with multiple covariates
"""


def linear_regression(
    x: Union[pd.DataFrame, np.ndarray], add_intercept=False, **constraints
) -> Kernel:
    r"""
    Linear regression of the columns in the data.

    .. math::

        y = \beta_0 + \sum_{i=1}^{n} \beta_i x_i

    Parameters
    ----------
    x : array-like or int
        The number of dimensions (int) or the data the kernel will be computed on.
        There will be one parameter for each column.
    add_intercept : bool
        If True, an intercept is added to the result.
    constraints : dict, optional
        Fixed values for the parameters of the regression. The constraints are given as
        ``beta_i=value``, where ``i`` is the index of the column starting with 1.
        If `x` is provided as a dataframe and the second column is named `cname`,
        the following constraints are equivalent: ``beta_2=2``, ``beta_cname=2``, ``cname=2``.
        The parameter ``beta_0`` constrains the value of the intercept if `add_intercept` is True.

    Returns
    -------
    float
        The linear sum computed from the input data.
    """
    if len(x.shape) > 1:
        ndim = x.shape[1]
    else:
        raise ValueError("Consider using kernels.linear for a 1-dimensional data array")
    fixed = {}
    for p_name, p_value in constraints.items():
        if p_name.startswith("beta_"):
            p_name = p_name[len("beta_") :]
        if isinstance(x, pd.DataFrame) and p_name in x.columns:
            index = tuple(x.columns).index(p_name) + 1
        else:
            index = int(p_name)
        fixed[index] = ensure_parametrized(p_value, constant=True)

    if 0 in fixed and not add_intercept:
        raise ValueError(
            "A fixed value is given for the intercept, but `add_intercept` is not True."
        )

    param_indices = range(0 if add_intercept else 1, ndim + 1)
    params = {
        f"beta_{i}": Parameter(0.0) if i not in fixed else fixed[i]
        for i in param_indices
    }

    def _compute(data, **params_from_wrapper):
        intercept = params_from_wrapper.pop("beta_0", 0)
        sorted_params = [params_from_wrapper[k] for k in params if k != "beta_0"]
        return intercept + (sorted_params * data).sum(axis=1)

    return Kernel(_compute, x, **params, fname=linear_regression.__qualname__)


def exponential_linear_regression(
    x: Union[pd.DataFrame, np.ndarray], add_intercept=False, **constraints
) -> Kernel:
    r"""
    Exponential of a linear sum of the columns in the data.

    .. math::

        y = \exp\left(\beta_0 + \sum_{i=1}^{n} \beta_i x_i\right)

    Parameters
    ----------
    x : array-like or int
        The number of dimensions (int) or the data the kernel will be computed on.
        There will be one parameter for each column.
    add_intercept : bool
        If True, an intercept is added to the result.
    constraints : dict, optional
        Fixed values for the parameters of the regression. The constraints are given as
        ``beta_i=value``, where ``i`` is the index of the column starting with 1.
        If `x` is provided as a dataframe and the second column is named `cname`,
        the following constraints are equivalent: ``beta_2=2``, ``beta_cname=2``, ``cname=2``.
        The parameter ``beta_0`` constrains the value of the intercept if `add_intercept` is True.

    Returns
    -------
    float
        The linear sum computed from the input data.
    """
    if len(x.shape) > 1:
        ndim = x.shape[1]
    else:
        raise ValueError("Consider using kernels.expo for a 1-dimensional data array")
    fixed = {}
    for p_name, p_value in constraints.items():
        if p_name.startswith("beta_"):
            p_name = p_name[len("beta_") :]
        if isinstance(x, pd.DataFrame) and p_name in x.columns:
            index = tuple(x.columns).index(p_name) + 1
        else:
            index = int(p_name)
        fixed[index] = ensure_parametrized(p_value, constant=True)

    if 0 in fixed and not add_intercept:
        raise ValueError(
            "A fixed value is given for the intercept, but `add_intercept` is not True."
        )

    param_indices = range(0 if add_intercept else 1, ndim + 1)
    params = {
        f"beta_{i}": Parameter(0.0) if i not in fixed else fixed[i]
        for i in param_indices
    }

    def _compute(data, **params_from_wrapper):
        sorted_params = np.stack([params_from_wrapper[k] for k in params])
        return np.exp((sorted_params * data).sum(axis=1))

    return Kernel(
        _compute, x, **params, fname=exponential_linear_regression.__qualname__
    )


def polynomial_regression(
    x: Union[pd.DataFrame, np.ndarray],
    degree: Union[int, Sequence] = 2,
    **constraints,
) -> Kernel:
    r"""
    Polynomial regression in the columns of the data.

    .. math::

        y = \sum_{i=1}^{n} \sum_{d=1}^{D_i} \beta_{i,d} x_i^d

    Parameters
    ----------
    x : array-like or int
        The number of dimensions (int) or the data the kernel will be computed on.
        There will be one parameter for each column.
    degree : int or Sequence
        The degree of the polynomial for each covariate. If an integer, the same degree is used for all.
    constraints : dict, optional
        Fixed values for the parameters of the regression. The constraints are given as
        ``beta_i_d=value``, where ``i`` is the index of the column starting with 1 and ``d`` is the degree.
        If `x` is provided as a dataframe and the second column is named `cname`,
        the following constraints are equivalent: ``beta_2_2=2``, ``beta_cname_2=2``, ``cname_2=2``.

    Returns
    -------
    float
        The polynomial regression computed from the input data.
    """
    if len(x.shape) > 1:
        ndim = x.shape[1]
    else:
        raise ValueError("Consider using kernels.linear for a 1-dimensional data array")
    if isinstance(degree, int):
        assert degree > 0, "This model considers positive power laws only."
        degree = [degree] * ndim
    else:
        assert len(degree) == ndim, (
            "The number of degrees is different than the number of covariates."
        )
    ncols = sum(degree)
    fixed = {}
    for p_name, p_value in constraints.items():
        if p_name.startswith("beta_"):
            p_name = p_name[len("beta_") :]
        match = re.match(r"^(.+)_(\d+)$", p_name)
        if match:
            column, deg = match.groups()
        else:
            raise ValueError(f"Unable to parse parameter constraint: {p_name}")
        if isinstance(x, pd.DataFrame) and column in x.columns:
            column = list(x.columns).index(column) + 1
        fixed[(int(column), int(deg))] = ensure_parametrized(p_value, constant=True)
    params = {}
    for col_idx, max_degree in enumerate(degree):
        for d in range(1, max_degree + 1):
            name = f"beta_{col_idx + 1}_{d}"
            params[name] = fixed.get((col_idx + 1, d), Parameter(0.0))

    def _compute(data, **params_from_wrapper):
        data = np.array(data)
        data_with_extra_cols = np.zeros(shape=(len(data), ncols))
        extra_col_idx = 0
        for col_idx, max_degree in enumerate(degree):
            for d in range(1, max_degree + 1):
                data_with_extra_cols[:, extra_col_idx] = data[:, col_idx] ** d
                extra_col_idx += 1
        sorted_params = [params_from_wrapper[k] for k in params]
        return (sorted_params * data_with_extra_cols).sum(axis=1)

    return Kernel(_compute, x, **params, fname=polynomial_regression.__qualname__)


def categories_qualitative(x: Collection, fixed_values: dict = None) -> Kernel:
    """
    Kernel for qualitative (categorical) data.

    Parameters
    ----------
    x : Collection
        The qualitative data containing categorical values (e.g., strings or integers).
    fixed_values : dict, optional
        A dictionary specifying constant values for certain categories. The keys are the category names,
        and the values are the fixed parameter values.

    Returns
    -------
    Kernel
        A kernel function that assigns a parameter to each unique value in the data.

    Notes
    -----
    The kernel creates one parameter for each unique category in the data. If `fixed_values` is provided,
    the corresponding categories will use the fixed parameters instead of creating new ones.

    Examples
    --------
    >>> data = ['A', 'B', 'A', 'C']
    >>> kernel = categories_qualitative(data, fixed_values={'A': 1.0})
    """
    unique_values = sorted(set(map(str, x)))
    fixed_values = {str(k): v for k, v in (fixed_values or {}).items()}
    parameter = (Parameter(0.0) for _ in count())  # generate parameters on demand
    params = {
        value: (
            next(parameter)
            if value not in fixed_values
            else ensure_parametrized(fixed_values[value], constant=True)
        )
        for value in unique_values
    }

    def _compute(data, **params_from_wrapper):
        return type(data)(list(map(lambda v: params_from_wrapper[str(v)], data)))

    return Kernel(_compute, x, **params, fname=categories_qualitative.__qualname__)
