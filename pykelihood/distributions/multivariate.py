from itertools import islice
from typing import Iterable, Mapping, Sequence, Tuple, Union

import copulae
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from pykelihood.distributions import Distribution
from pykelihood.parameters import ConstantParameter, Parameter, Parametrized


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_squared_and_symmetric(x):
    if x.shape[0] != x.shape[1]:
        return False
    return np.all(x == x.T)


def build_parameter_matrix(rows, cols, fixed_values):
    res = []
    for row in range(1, rows + 1):
        res.append([])
        for col in range(1, cols + 1):
            index = (row, col)
            if index in fixed_values:
                res[-1].append(ConstantParameter(fixed_values[index]))
            else:
                res[-1].append(Parameter())
    return res


def build_parameter_vector(size, fixed_values):
    res = []
    for index in range(1, size + 1):
        if index in fixed_values:
            res.append(ConstantParameter(fixed_values[index]))
        else:
            res.append(Parameter())
    return res


def build_parameter_array(shape, fixed_values):
    if len(shape) > 2:
        raise ValueError("Dimensions higher than 2 are not supported")
    try:
        n, m = shape
    except ValueError:
        (n,) = shape
        return build_parameter_vector(n, fixed_values)
    return build_parameter_matrix(n, m, fixed_values)


def flatten(array):
    res = []
    for x in array:
        if isinstance(x, Iterable):
            res.extend(x)
        else:
            res.append(x)
    return res


class ParameterArray(Parametrized):
    def __init__(
        self,
        shape: Union[int, Tuple[int, int]],
        fixed_values: Mapping[Tuple[int, int], float] = None,
    ):
        # create array of parameter objects
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self._params = self._create_params(shape, fixed_values)
        self._matrix = self._as_matrix()
        # self.shape = len(self._params), len(self._params[-1])
        super(ParameterArray, self).__init__(*flatten(self._params))

    def _create_params(self, shape, fixed_values):
        return build_parameter_array(shape, fixed_values)

    def with_params(self, params):
        params = iter(params)
        my_values = list(islice(params, self.size))
        new_value = np.array(my_values).reshape(self.value.shape)
        return ParameterArray(new_value)

    def _as_matrix(self):
        return self._params

    def __getitem__(self, item):
        return self._matrix[item]

    def __call__(self):
        return np.asarray(self._matrix).reshape(self.shape)

    def __len__(self):
        return len(self.value)


def get_both_or_one_of(d, k1, k2, default=None):
    if k1 in d and k2 in d:
        if d[k1] != d[k2]:
            raise ValueError(
                f"Unexpected difference at indices {k1} and {k2}: {d[k1]} vs {d[k2]}"
            )
        return d[k1]
    if k1 in d:
        return d[k1]
    if k2 in d:
        return d[k2]
    return default


class ParameterCorrelationMatrix(ParameterArray):
    def __init__(self, width: int, fixed_values):
        self.width = width
        super(ParameterCorrelationMatrix, self).__init__((width, width), fixed_values)

    def _create_params(self, shape, fixed_values) -> Sequence[Parameter]:
        if not isinstance(shape, int):
            if shape[0] != shape[1]:
                raise ValueError("Expected a square matrix.")
            shape = shape[0]
        res = []
        for row in range(1, shape + 1):
            for col in range(row + 1, shape + 1):
                v = get_both_or_one_of(fixed_values, (row, col), (col, row))
                if v is not None:
                    res.append(ConstantParameter(v))
                else:
                    res.append(Parameter())
        return res

    def _as_matrix(self):
        params = iter(self.params)
        res = [[0] * self.width for _ in range(self.width)]
        for row in range(self.width):
            res[row][row] = 1
            for col in range(row + 1, self.width):
                v = float(next(params))
                res[row][col] = res[col][row] = v
        return res

    @property
    def valid(self):
        return (
            (is_pos_def(self.value))
            and (is_squared_and_symmetric(self.value))
            and np.all((self.value <= 1) & (self.value >= -1))
        )

    def with_params(self, params):
        new_corr_matrix = np.eye(self.size)
        params = iter(params)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                param = next(params)
                new_corr_matrix[i, j] = new_corr_matrix[j, i] = param
        return new_corr_matrix


class Copula(object):
    def __init__(self, margins_distribution: Distribution, correlation_structure):
        self.margins_distribution = margins_distribution
        self.correlation_structure = correlation_structure

    def fit(self, data):
        new_corr_structure = self.correlation_structure.fit(data)
        return type(self)(self.margins_distribution, new_corr_structure)

    def fit2(self, data):
        num_params = self.correlation_structure.num_params()

        def to_minimize(x):
            new_structure = self.correlation_structure.with_params(x)
            if not new_structure.valid:
                return 10 ** 10
            pdf = new_structure.pdf(data)
            return -np.sum(np.log(pdf))

        res = minimize(
            to_minimize, x0=np.array([0] * num_params), options={"maxiter": 10000}
        )
        final_corr = self.correlation_structure.with_params(res.x)
        return Copula(self.margins_distribution, final_corr)


class CorrelationStructure(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, x, *args, **kwargs):
        self.func(x, *args, **kwargs)

    def extremal_correlation(self):
        pass


class Logistic(object):
    def __init__(self, alpha):
        if (alpha <= 0) or (alpha > 1):
            raise ValueError(
                r"The parameter \alpha of the Logistic copula takes values in (0, 1]."
            )
        self.alpha = alpha

    def cdf(self, x):
        return np.exp(-((np.sum(x ** (-1 / self.alpha))) ** self.alpha))

    def extremal_correlation(self):
        return 2 * (1 - 2 ** (self.alpha - 1))

    def extremal_independance(self):
        pass


class Gaussian(object):
    copulae_equivalent = copulae.GaussianCopula
    params_names = ("mean", "cov")

    def __init__(self, dim, cov=None):
        self.dim = dim
        self.mean = np.array([0] * self.dim)
        self.cov = ParameterCorrelationMatrix(
            cov if cov is not None else np.eye(dim, dim)
        )

    @property
    def valid(self):
        return self.cov.valid

    def cdf(self, x):
        return multivariate_normal.cdf(x, self.mean, self.cov.value)

    def pdf(self, x):
        return multivariate_normal.pdf(x, self.mean, self.cov.value)

    def num_params(self):
        return self.dim * (self.dim - 1) // 2

    @classmethod
    def from_params(cls, cov):
        dim = len(cov)
        return cls(dim, cov)

    def with_params(self, params):
        params = iter(params)
        cov = self.cov.with_params(params)
        return Gaussian.from_params(cov)

    def fit(self, data):
        cop_obj = self.copulae_equivalent(self.dim)
        cop_obj.fit(
            data,
        )
        return type(self)(self.dim, cov=cop_obj.sigma)


if __name__ == "__main__":
    import univariate

    margin = univariate.Normal()
    # c = Copula(margin, correlation_structure=Gaussian(3))
    # data = np.array([univariate.Normal().rvs(10000) for _ in range(3)])
    # corr = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
    # print(corr)
    # data = multivariate_normal.rvs([0, 0, 0], cov=corr, size=10000)
    # c1 = c.fit(data)
    # print(c1.correlation_structure.cov.value)
    # c2 = c.fit2(data)
    # print(c2.correlation_structure.cov.value)

    m = ParameterArray((3, 2), fixed_values={(1, 2): 3})
    print(m())

    c = ParameterCorrelationMatrix(4, {(3, 1): 2})
    print(c())
