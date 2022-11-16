import re
from itertools import count
from typing import Collection, Sequence, Union

import numpy as np
import pandas as pd

from pykelihood.parameters import Parameter, ParametrizedFunction, ensure_parametrized


class Kernel(ParametrizedFunction):
    def __init__(self, f, x=None, fname=None, **params):
        """Function of one covariate with parameters."""
        fname = fname or f.__qualname__
        super(Kernel, self).__init__(f, fname=fname, **params)
        self._validate(x)
        self.x = x

    def _validate(self, x):
        try:
            iter(x)
        except TypeError:
            raise ValueError(
                f"Incorrect covariate for {self.fname}: {self.x}"
            ) from None

    def __call__(self, x=None):
        if x is None:
            x = self.x
        self._validate(x)
        param_values = {p_name: p() for p_name, p in self.param_dict.items()}
        return self.f(x, **param_values)

    def _build_instance(self, **new_params):
        return type(self)(self.f, self.x, fname=self.fname, **new_params)

    def with_covariate(self, covariate):
        return type(self)(self.f, covariate, fname=self.fname, **self.param_dict)


class constant(Kernel):
    def __init__(self, value=0.0):
        super(constant, self).__init__(self._call, fname="constant", value=value)

    def _build_instance(self, value):
        return constant(value)

    def _validate(self, x):
        if x is not None:
            raise ValueError(f"Unexpected data for constant kernel: {x}")

    def _call(self, _x, value):
        return value


def kernel(**param_defaults):
    def wrapper(f):
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
    return a + b * X


@kernel(a=0.0, b=0.0, c=0.0)
def polynomial(X, a, b, c):
    return a + b * X + c * X**2


@kernel(a=0.0, b=0.0)
def expo(X, a, b):
    inner = b * X
    inner = a + inner
    return np.exp(inner)


@kernel(a=0.0, b=1.0, c=1.0)
def expo_ratio(X, a, b, c):
    inner = a * X
    inner = inner / b
    return c * np.exp(inner)


@kernel(mu=0.0, sigma=1.0, scaling=0.0)
def gaussian(X, mu, sigma, scaling):
    mult = scaling * 1 / (sigma * np.sqrt(2 * np.pi))
    expo = np.exp(-((X - mu) ** 2) / sigma**2)
    return mult * expo


@kernel(a=0.0, b=0.0, c=0.0)
def trigonometric(X, a, b, c):
    """

    :param X: vector rescaled per period of interest. Example: we are interested in seasonal variations for daily observations over 10 years. X can be
    the number of days since start, on which we apply lambda x: x//365.25, and then groupby integer values (years) and rescale each time vector per year between 0 and 1.
    This way, we get yearly aligned periods corresponding to seasons.
    :return: a simple trigonometric kernel
    """
    return a + b * np.cos(2 * np.pi * X) + c * np.sin(2 * np.pi * X)


@kernel(mu=0.0, alpha=0.0, theta=1.0)
def hawkes_with_exp_kernel(X, mu, alpha, theta):
    return mu + alpha * theta * np.array(
        [np.sum(np.exp(-theta * (X[i] - X[:i]))) for i in range(len(X))]
    )


def hawkes2(t, tau, mu, alpha, theta):
    return mu + alpha * theta * np.sum(
        (np.exp(-theta * (t - tau[i]))) for i in range(len(tau)) if tau[i] < t
    )


"""
Sophisticated kernels with multiple covariates
"""


def linear_regression(
    x: Union[pd.DataFrame, np.ndarray], add_intercept=False, **constraints
) -> Kernel:
    """Computes a trend as a linear sum of the columns in the data.

    :param x: the data the kernel will be computed on. There will be one parameter for each column.
    :param add_intercept: if True, an intercept is added to the result
    :param constraints: fixed values for the parameters of the regression. The constraints are given as `beta_i=value`,
                        where i is the index of the column starting with 1.
                        If data is given as a dataframe with the second column named 'cname', the following constraints are equivalent:
                        'beta_2=2', 'beta_cname=2', 'cname=2'.
                        'beta_0' constrains the value of the intercept if add_intercept is True.
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
        f"beta_{i}": Parameter() if i not in fixed else fixed[i] for i in param_indices
    }

    def _compute(data, **params_from_wrapper):
        intercept = params_from_wrapper.pop("beta_0", 0)
        sorted_params = [params_from_wrapper[k] for k in params if k != "beta_0"]
        return intercept + (sorted_params * data).sum(axis=1)

    return Kernel(_compute, x, **params, fname=linear_regression.__qualname__)


def exponential_linear_regression(
    x: Union[pd.DataFrame, np.ndarray], add_intercept=False, **constraints
) -> Kernel:
    """Computes a trend as the exponential of a linear sum of the columns in the data.

    :param x: the number of dimensions or the data the kernel will be computed on. There will be one parameter for each column.
    :param add_intercept: if True, an intercept is added to the result
    :param constraints: fixed values for the parameters of the regression. The constraints are given as `beta_i=value`,
                        where i is the index of the column starting with 1.
                        If data is given as a dataframe with the second column named 'cname', the following constraints are equivalent:
                        'beta_2=2', 'beta_cname=2', 'cname=2'.
                        'beta_0' constrains the value of the intercept if add_intercept is True.
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
        f"beta_{i}": Parameter() if i not in fixed else fixed[i] for i in param_indices
    }

    def _compute(data, **params_from_wrapper):
        sorted_params = [params_from_wrapper[k] for k in params]
        return np.exp((sorted_params * data).sum(axis=1))

    return Kernel(
        _compute, x, **params, fname=exponential_linear_regression.__qualname__
    )


def polynomial_regression(
    x: Union[pd.DataFrame, np.ndarray],
    degree: Union[int, Sequence] = 2,
    **constraints,
) -> Kernel:
    """Computes a trend as the sum of the columns in the data to the power of n for n smaller or equal to degree.

    :param x: the number of dimensions or the data the kernel will be computed on. There will be one parameter for each column.
    :param degree: last exponent computed for the given covariates. Can be a list or np array, but if this is the case, the number of
    exponents should be equal to the number of columns of x.
    :param constraints: fixed values for the parameters of the regression. The following constraints are equivalent:
                        'beta_2_2=2', 'beta_cname_2=2', 'cname_2=2'
                        The last two are valid only if data is given as a dataframe with the second column named 'cname'.
    """
    if len(x.shape) > 1:
        ndim = x.shape[1]
    else:
        raise ValueError("Consider using kernels.linear for a 1-dimensional data array")
    if isinstance(degree, int):
        assert degree > 0, "This model considers positive power laws only."
        degree = [degree] * ndim
    else:
        assert (
            len(degree) == ndim
        ), "The number of degrees is different than the number of covariates."
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
            params[name] = fixed.get((col_idx + 1, d), Parameter())

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
    unique_values = sorted(set(map(str, x)))
    fixed_values = {str(k): v for k, v in (fixed_values or {}).items()}
    parameter = (Parameter() for _ in count())  # generate parameters on demand
    params = {
        value: next(parameter)
        if value not in fixed_values
        else ensure_parametrized(fixed_values[value], constant=True)
        for value in unique_values
    }

    def _compute(data, **params_from_wrapper):
        return type(data)(list(map(lambda v: params_from_wrapper[str(v)], data)))

    return Kernel(_compute, x, **params, fname=categories_qualitative.__qualname__)
