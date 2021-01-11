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
        (regression.with_params([1, 2, 3])(matrix_data) == [1, 2, 3] * matrix_data)
        .all()
        .all()
    )


def test_linear_regression_with_data(matrix_data):
    regression = kernels.linear_regression(matrix_data)
    assert len(regression.params) == len(matrix_data.columns)
    assert (regression.with_params([1, 2, 3])() == [1, 2, 3] * matrix_data).all().all()


def test_linear_regression_with_constraint(matrix_data):
    regression = kernels.linear_regression(matrix_data, beta_1=5, _0=6)
    assert len(regression.optimisation_params) == 1
    assert (regression.with_params([1])() == [6, 5, 1] * matrix_data).all().all()


def test_linear_regression_with_name_constraint(matrix_data):
    regression = kernels.linear_regression(matrix_data, first=5, beta_third=2)
    assert len(regression.optimisation_params) == 1
    assert (regression.with_params([3])() == [5, 3, 2] * matrix_data).all().all()
