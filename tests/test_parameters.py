import pytest

from pykelihood import parameters


def test_parameter():
    p = parameters.Parameter(3.0)
    assert p() == 3.0


def test_parameter_with_params():
    p = parameters.Parameter()
    assert p.with_params([5.0])() == 5.0


def test_flattened_params():
    p1 = parameters.Parameter()
    p2 = parameters.ConstantParameter()
    a = parameters.Parametrized(p1, 2)
    a.params_names = ("x", "m")
    repr(a)
    b = parameters.Parametrized(a, p2)
    b.params_names = ("y", "n")
    assert set(a.flattened_param_dict.keys()) == {"x", "m"}
    assert len(a.flattened_params) == 2
    assert set(b.flattened_param_dict.keys()) == {"y_x", "y_m", "n"}
    assert len(b.flattened_params) == 3


def test_flattened_params_with_embedded_constant():
    p1 = parameters.Parameter()
    p2 = parameters.ConstantParameter()
    a = parameters.Parametrized(p1, p2)
    a.params_names = ("x", "m")
    b = parameters.Parametrized(a)
    b.params_names = "y"
    assert set(a.flattened_param_dict.keys()) == {"x", "m"}
    assert len(a.flattened_params) == 2
    assert set(b.flattened_param_dict.keys()) == {"y_x", "y_m"}
    assert len(b.flattened_params) == 2
    assert set(b.optimisation_param_dict.keys()) == {"y_x"}
    assert len(b.optimisation_params) == 1


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

    def test_named_with_params(self, func):
        f = parameters.ParametrizedFunction(func, p=parameters.Parameter(5))
        modf = f.with_params(p=2)
        assert modf(2) == 4

    def test_func_name(self, func):
        p = parameters.Parameter(1.0)
        f = parameters.ParametrizedFunction(func, fname="myfunc", p=p)
        assert repr(f) == "myfunc(p=1.0)"


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

    def test_parameter_optimisation_param_dict(self, func):
        p1 = parameters.Parameter(2)
        p2 = parameters.ParametrizedFunction(func, p=p1)
        assert p2.optimisation_param_dict == {"p": p1}

    def test_parameter_optimisation_param_dict_multi_level(self, func):
        p1 = parameters.Parameter(2)
        p2 = parameters.ParametrizedFunction(func, p=p1)
        p3 = parameters.ParametrizedFunction(func, p=p2)
        assert p3.optimisation_param_dict == {"p_p": p1}

    def test_parameter_optimisation_param_dict_constant(self, func):
        p1 = parameters.ConstantParameter(2)
        p2 = parameters.ParametrizedFunction(func, p=p1)
        p3 = parameters.ParametrizedFunction(func, p=p2)
        assert p3.optimisation_param_dict == {}

    def test_named_with_params_multi_level(self, func):
        p1 = parameters.Parameter(1)
        q = parameters.Parameter(2)
        p2 = parameters.ParametrizedFunction(func, p=p1)
        p3 = parameters.ParametrizedFunction(func, p=p2)
        assert p3.with_params(p_p=q).optimisation_param_dict == {"p_p": q}

    def test_named_with_params_partial_assignment(self, func):
        p = parameters.Parameter(2)
        q = parameters.Parameter(3)
        x = parameters.Parameter(4)
        f = parameters.ParametrizedFunction(func, p=p, x=x)
        assert f() == 6
        fmod = f.with_params(p=q)
        assert fmod.x is x
        assert fmod.p is q
        assert fmod() == 7

    def test_named_with_params_nested_replacement(self, func):
        pf = parameters.ParametrizedFunction(func, p=parameters.Parameter(1))
        pf2 = parameters.ParametrizedFunction(func, p=pf)
        q = parameters.Parameter(2)
        assert pf2.with_params(p=q).optimisation_param_dict == {"p": q}
        assert pf2.with_params(p_p=q).optimisation_param_dict == {"p_p": q}
