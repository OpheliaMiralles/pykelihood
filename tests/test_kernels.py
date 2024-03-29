from itertools import count

import numpy as np

from pykelihood import kernels, parameters


def test_constant_kernel():
    constant = kernels.constant(2)
    assert constant() == 2
    assert constant.with_params(value=5)() == 5


def test_kernel():
    x = np.array([1, 2, 3])
    linear = kernels.linear(x, a=0, b=2)
    assert list(linear()) == list(2 * x)
    assert list(linear.with_covariate(1 + x)()) == list(2 * (1 + x))
    assert list(linear()) == list(2 * x)


def test_linear_trend(dataset):
    trend = kernels.linear(dataset)
    assert len(trend.params) == 2
    assert len(trend.optimisation_params) == 2
    assert (trend.with_params([3, 4])() == 3 + dataset * 4).all()


def test_linear_trend_with_constraint(dataset):
    trend = kernels.linear(dataset, a=2)
    assert len(trend.params) == 2
    assert len(trend.optimisation_params) == 1
    assert trend.a() == 2
    assert (trend.with_params([3])() == 2 + dataset * 3).all()


def test_trend_in_trend(dataset):
    inner_trend = kernels.linear(dataset)
    outer_trend = kernels.linear(dataset, b=inner_trend)
    assert {"a", "b_a", "b_b"} == set(outer_trend.flattened_param_dict.keys())


def test_linear_regression(matrix_data):
    regression = kernels.linear_regression(matrix_data)
    assert len(regression.params) == len(matrix_data.columns)
    assert (
        regression.with_params([1, 2, 3])() == ([1, 2, 3] * matrix_data).sum(axis=1)
    ).all()


def test_linear_regression_with_constraint(matrix_data):
    regression = kernels.linear_regression(matrix_data, beta_2=5, beta_1=6)
    assert len(regression.optimisation_params) == 1
    assert (
        regression.with_params([1])() == ([6, 5, 1] * matrix_data).sum(axis=1)
    ).all()


def test_linear_regression_with_name_constraint(matrix_data):
    regression = kernels.linear_regression(matrix_data, first=5, beta_third=2)
    assert len(regression.optimisation_params) == 1
    assert (
        regression.with_params([3])() == ([5, 3, 2] * matrix_data).sum(axis=1)
    ).all()


def test_linear_regression_with_name_constraint_parameter(matrix_data):
    p = parameters.Parameter(5)
    regression = kernels.linear_regression(matrix_data, first=p, beta_third=2)
    assert len(regression.optimisation_params) == 2
    assert len(regression.flattened_params) == 3


def test_linear_regression_with_intercept(matrix_data):
    regression = kernels.linear_regression(
        matrix_data, first=5, beta_third=2, add_intercept=True
    )
    assert len(regression.optimisation_params) == 2
    assert (
        regression.with_params([10, 3])() == 10 + ([5, 3, 2] * matrix_data).sum(axis=1)
    ).all()


def test_exponential_linear_regression_with_name_constraint(matrix_data):
    regression = kernels.exponential_linear_regression(matrix_data, beta_2=5, beta_1=6)
    assert len(regression.optimisation_params) == 1
    assert (
        regression.with_params([1])() == np.exp(([6, 5, 1] * matrix_data).sum(axis=1))
    ).all()


def test_polynomial_regression_with_uniform_degree_across_columns(matrix_data):
    regression = kernels.polynomial_regression(matrix_data, degree=2)
    nparams = 2 * len(matrix_data.columns)
    assert len(regression.optimisation_params) == nparams
    np.testing.assert_allclose(
        regression.with_params([1] * nparams)(),
        matrix_data.sum(axis=1) + (matrix_data**2).sum(axis=1),
    )


def test_polynomial_regression_with_different_degrees(matrix_data):
    matrix_data = matrix_data.drop(columns="third")
    degrees = [2, 1]
    regression = kernels.polynomial_regression(matrix_data, degree=degrees)
    nparams = sum(degrees)
    assert len(regression.optimisation_params) == nparams
    np.testing.assert_allclose(
        regression.with_params([1] * nparams)(),
        (matrix_data).sum(axis=1) + matrix_data["first"] ** 2,
    )


def test_polynomial_regression_with_constraint(matrix_data):
    matrix_data = matrix_data.drop(columns="third")
    degrees = [2, 1]
    regression = kernels.polynomial_regression(matrix_data, degree=degrees, beta_1_1=2)
    nparams = sum(degrees) - 1
    assert len(regression.optimisation_params) == nparams
    np.testing.assert_allclose(
        regression.with_params([1] * nparams)(),
        2 * matrix_data["first"] + matrix_data["first"] ** 2 + matrix_data["second"],
    )


def test_polynomial_regression_with_name_constraint(matrix_data):
    matrix_data = matrix_data.drop(columns="third")
    degrees = [2, 1]
    regression = kernels.polynomial_regression(matrix_data, degree=degrees, first_1=2)
    nparams = sum(degrees) - 1
    assert len(regression.optimisation_params) == nparams
    np.testing.assert_allclose(
        regression.with_params([1] * nparams)(),
        2 * matrix_data["first"] + matrix_data["first"] ** 2 + matrix_data["second"],
    )


def test_categorical(categorical_data):
    cat_kernel = kernels.categories_qualitative(categorical_data)
    assert len(cat_kernel.optimisation_params) == categorical_data.unique().size
    mapping = {k: v for k, v in zip(cat_kernel.params_names, count(1))}
    # the values are "item1", "item2" or "item3"
    assert (
        cat_kernel.with_params(mapping.values())()
        == categorical_data.apply(lambda x: mapping[x])
    ).all()


def test_categorical_with_constraint(categorical_data):
    cat_kernel = kernels.categories_qualitative(
        categorical_data, dict(item1=1, item2=8)
    )
    assert len(cat_kernel.optimisation_params) == categorical_data.unique().size - 2
    mapping = {"item1": 1, "item2": 8, "item3": 12}
    assert (
        cat_kernel.with_params([12])() == categorical_data.apply(mapping.__getitem__)
    ).all()


def test_categorical_with_bool(categorical_data_boolean):
    cat_kernel = kernels.categories_qualitative(categorical_data_boolean)
    assert len(cat_kernel.optimisation_params) == 2
    mapping = {True: 1, False: 8}
    values = [
        next(v for k, v in mapping.items() if str(k) == p_name)
        for p_name in cat_kernel.params_names
    ]
    assert len(values) == 2
    assert (
        cat_kernel.with_params(values)()
        == categorical_data_boolean.apply(mapping.__getitem__)
    ).all()
