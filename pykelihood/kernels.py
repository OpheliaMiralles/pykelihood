import numpy as np

from pykelihood.parameters import ParametrizedFunction, Parameter, ConstantParameter


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

@parametrized_function(a=0., b=0., c=0.)
def three_categories_qualitative(X, a, b, c):
    mapping_cats = {cat : factor for cat, factor in zip(list(sorted(X.unique())), [a, b, c])}
    return X.apply(lambda x: mapping_cats[x])


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





