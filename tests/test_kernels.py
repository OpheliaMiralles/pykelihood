from itertools import count

from pykelihood import kernels


def test_linear_trend(dataset):
    trend = kernels.linear(dataset)
    assert len(trend.params) == 2
    assert len(trend.optimisation_params) == 2
    assert (trend.with_params([3, 4])() == 3 + dataset * 4).all()


def test_linear_trend_with_constraint(dataset):
    trend = kernels.linear(dataset, a=2)
    assert len(trend.params) == 2
    assert len(trend.optimisation_params) == 1
    assert trend.a == 2
    assert (trend.with_params([3])() == 2 + dataset * 3).all()


def test_linear_trend_deferred_data(dataset):
    trend = kernels.linear()
    assert len(trend.params) == 2
    assert len(trend.optimisation_params) == 2
    assert (trend.with_params([3, 4])(dataset) == 3 + dataset * 4).all()


def test_linear_regression(matrix_data):
    regression = kernels.linear_regression(3)
    assert len(regression.optimisation_params) == 3
    assert (
        regression.with_params([1, 2, 3])(matrix_data)
        == ([1, 2, 3] * matrix_data).sum(axis=1)
    ).all()


def test_linear_regression_with_data(matrix_data):
    regression = kernels.linear_regression(matrix_data)
    assert len(regression.params) == len(matrix_data.columns)
    assert (
        regression.with_params([1, 2, 3])() == ([1, 2, 3] * matrix_data).sum(axis=1)
    ).all()


def test_linear_regression_with_constraint(matrix_data):
    regression = kernels.linear_regression(matrix_data, beta_1=5, _0=6)
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
    cat_kernel = kernels.categories_qualitative(categorical_data, item1=1, item2=8)
    assert len(cat_kernel.optimisation_params) == categorical_data.unique().size - 2
    mapping = {"item1": 1, "item2": 8, "item3": 12}
    assert (
        cat_kernel.with_params([12])() == categorical_data.apply(mapping.__getitem__)
    ).all()
