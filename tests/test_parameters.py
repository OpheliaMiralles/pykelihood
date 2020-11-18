import pytest

from pykelihood import parameters


class TestParameter:
    def test_parameter(self):
        p = parameters.Parameter(3.)
        assert p == 3.

    def test_parameter_with_params(self):
        p = parameters.Parameter()
        assert p.with_params([5.]) == 5.


@pytest.fixture(scope="module")
def func():
    def f(x, p):
        return x + p
    return f


class TestParametrizedFunction:
    def test_parameters(self, func):
        f = parameters.ParametrizedFunction(func, p=parameters.Parameter(5))
        assert f(2) == 7

    def test_with_params(self, func):
        f = parameters.ParametrizedFunction(func, p=parameters.Parameter(5))
        modf = f.with_params([2])
        assert modf(2) == 4

    def test_attr_access(self, func):
        p = parameters.Parameter()
        f = parameters.ParametrizedFunction(func, p=p)
        assert f.p is p


class TestInnerParameters:
    def test_function_parameters(self, func):
        p = parameters.Parameter(5)
        f = parameters.ParametrizedFunction(func, p=p)
        assert len(f.params) == 1
        assert f.params[0] is p
        assert len(f.optimisation_params) == 1
        assert f.optimisation_params[0] is p

    def test_function_constants(self, func):
        p = parameters.ConstantParameter(5)
        f = parameters.ParametrizedFunction(func, p=p)
        assert len(f.params) == 1
        assert f.params[0] is p
        assert len(f.optimisation_params) == 0

    def test_multiple_parameters(self, func):
        p1 = parameters.Parameter(2)
        p2 = parameters.ConstantParameter(3)
        f = parameters.ParametrizedFunction(func, x=p1, p=p2)
        assert f() == 5
        assert len(f.params) == 2
        assert len(f.optimisation_params) == 1
        assert f.with_params([4])() == 7
