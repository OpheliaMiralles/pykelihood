from typing import Callable

import numpy as np

from pykelihood.distributions import Exponential, Uniform


def PoissonByThinning(T, lambd, M):
    """
    :param T: Maximum time
    :param lambd: Function or float describing the intensity of the Poisson Process
    :param M: Upper bound on [0, T] for lambd(.)
    :return: sample timestamps in ascending order
    """
    if not isinstance(lambd, Callable):
        functional_lambd = lambda x: lambd
    else:
        functional_lambd = lambd
    P = []
    t = 0
    while t < T:
        e = Exponential(loc=0.0, rate=M).rvs(1)
        t += e
        u = Uniform(loc=0.0, scale=M).rvs(1)
        if t < T and u <= functional_lambd(t):
            P.append(float(t))
    return P


def HawkesByThinning(T, lambd, tau=None):
    """
    From article https://arxiv.org/pdf/1507.02822.pdf
    :param T: Maximum time
    :param lambd: Intensity function, must be non-decreasing inbetween arrival times
    :param tau: true inter-arrival times, optional, if not provided the simulated ones are used
    :return: sample timestamps in ascending order
    """
    epsilon = 1e-10
    P = []
    tau = P if tau is None else tau
    t = 0
    while t < T:
        M = lambd(t + epsilon, tau)
        e = Exponential(rate=M).rvs(1)
        t += e
        u = Uniform(scale=M).rvs(1)
        if t < T and u <= lambd(t, tau):
            P.append(float(t))
    return P


def HawkesByThinningModified(T, mu, alpha, theta, phi=0):
    """
    Implements Ogata's modified thinning algorithm for sampling from a univariate Hawkes process
    with exponential decay.
    :param T: the maximum time
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param theta: intensity parameter of the delay density
    :param phi: (optionally) the starting phi, the running sum of exponential differences specifying the history until
    a certain time, thus making it possible to take conditional samples
    :return: 1-d numpy ndarray of samples
    """
    t = 0.0
    d = 0.0
    P = []
    while t < T:
        M = mu + alpha * theta * (1 + phi)
        e = Exponential(rate=M).rvs(1)
        t += e
        exp_decay = np.exp(-theta * (e + d))
        lambda_ = mu + alpha * theta * exp_decay * (1 + phi)
        u = Uniform(loc=0.0, scale=M).rvs(1)
        if t < T and u <= lambda_:
            P.append(float(t))
            phi = exp_decay * (1 + phi)
            d = 0
        else:
            d += e
    return P
