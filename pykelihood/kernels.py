from itertools import count
from typing import Collection, Union

import numpy as np
import pandas as pd

from pykelihood.parameters import ConstantParameter, Parameter, ParametrizedFunction


def parametrized_function(**param_defaults):
    def wrapper(f):
        def wrapped(*args, **param_values):
            final_params = {p_name: Parameter(v) for p_name, v in param_defaults.items()}
            final_params.update({p_name: ConstantParameter(v) for p_name, v in param_values.items()})
            return ParametrizedFunction(f, *args, **final_params)
        return wrapped
    return wrapper

@parametrized_function(a=0., b=0.)
def linear(X, a, b):
    return a+b*X


def linear_regression(x: Union[int, pd.DataFrame, np.ndarray] = 2, **constraints) -> ParametrizedFunction:
    """ Computes a trend as a linear sum of the columns in the data.

    :param x: the number of dimensions or the data the kernel will be computed on. There will be one parameter for each column.
    :param constraints: fixed values for the parameters of the regression. The following constraints are equivalent:
                        'beta_1=2', '_1=2', 'beta_cname=2', 'cname=2'
                        The last two are valid only if data is given as a dataframe with the second column named 'cname'.
    """
    args = ()
    if isinstance(x, int):
        assert x > 0, "Unexpected number of parameters for linear regression"
        ndim = x
    else:
        args = x,
        if len(x.shape) > 1:
            ndim = x.shape[1]
        else:
            raise ValueError("Consider using kernels.linear for a 1-dimensional data array")
    fixed = {}
    for p_name, p_value in constraints.items():
        if p_name.startswith("beta_"):
            p_name = p_name[len("beta_"):]
        if isinstance(x, pd.DataFrame) and p_name in x.columns:
            index = list(x.columns).index(p_name)
        else:
            if p_name.startswith("_"):
                p_name = p_name[1:]
            index = int(p_name)
        fixed[index] = ConstantParameter(p_value)
    params = {f"beta_{i}": Parameter() if i not in fixed else fixed[i] for i in range(ndim)}

    def _compute(data, **params_from_wrapper):
        sorted_params = [params_from_wrapper[k] for k in params]
        return (sorted_params * data).sum(axis=1)

    return ParametrizedFunction(_compute, *args, **params)


def categories_qualitative(x: Collection, fixed_values: dict = None) -> ParametrizedFunction:
    unique_values = sorted(set(map(str, x)))
    fixed_values = {str(k): v for k, v in (fixed_values or {}).items()}
    parameter = (Parameter() for _ in count())  # generate parameters on demand
    params = {
        value: next(parameter)
        if value not in fixed_values
        else ConstantParameter(fixed_values[value])
        for value in unique_values
    }

    def _compute(data, **params_from_wrapper):
        return type(data)(list(map(lambda v: params_from_wrapper[str(v)], data)))

    return ParametrizedFunction(_compute, x, **params)


@parametrized_function(a=0., b=0., c=0.)
def polynomial(X, a, b, c):
    return a+b*X+c*X**2

@parametrized_function(a=0., b=0., c=0.)
def trigo(X, a, b, c):
    return a + np.sum([b*np.cos(2*np.pi*l*X/365.) \
                       + c*np.sin(2*np.pi*l*X/365.) for l in range(len(X))])

@parametrized_function(a=0., b=0.)
def expo(X, a, b):
    inner = b * X
    inner = a + inner
    return np.exp(inner)

@parametrized_function(mu=0., sigma=1., scaling=0.)
def gaussian(X, mu, sigma, scaling):
    mult = scaling*1/(sigma*np.sqrt(2*np.pi))
    expo = np.exp(-(X-mu)**2/sigma**2)
    return mult*expo

@parametrized_function(mu=0., alpha=0., theta=1.)
def hawkes_with_exp_kernel(X, mu, alpha, theta):
    return mu + alpha * theta * np.array([np.sum(np.exp(-theta*(X[i]-X[:i]))) for i in range(len(X))])

def hawkes2(t, tau, mu, alpha, theta):
    return mu + alpha * theta * np.sum((np.exp(-theta * (t - tau[i]))) for i in range(len(tau)) if tau[i] < t)





