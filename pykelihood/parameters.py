from functools import partial
from typing import Tuple, Callable, Dict, Union, Iterable


class Parametrized(object):
    params_names: Tuple[str]

    def __init__(self, *params: Union['Parametrized', float]):
        self._params = tuple(Parameter(p) if not isinstance(p, Parametrized) else p for p in params)

    @property
    def params(self):
        return tuple(p_ for p in self._params for p_ in p.params)

    @property
    def param_dict(self) -> Dict[str, 'Parametrized']:
        return dict(zip(self.params_names, self._params))

    @property
    def names_and_params(self) -> Iterable[Tuple['Parametrized', str]]:
        for p, name in zip(self._params, self.params_names):
            if p.params:
                yield name, p

    def __repr__(self):
        args = [f"{a}={v!r}" for a, v in zip(self.params_names, self._params)]
        return f"{type(self).__name__}({', '.join(args)})"

    def with_params(self, params):
        params = iter(params)
        new_params = []
        for param in self._params:
            new_params.append(param.with_params(params))
        return type(self)(*new_params)

    def __getattr__(self, param):
        param = param
        try:
            idx = self.params_names.index(param)
        except ValueError:
            raise AttributeError(f"No parameter {param} in {type(self).__name__}")
        return self._params[idx]


class Parameter(float, Parametrized):

    def __new__(cls, x):
        return float.__new__(cls, x)

    def __init__(self, *args, **kwargs):
        pass

    @property
    def params(self):
        return self,

    def with_params(self, params):
        param = next(iter(params))
        if isinstance(param, ConstantParameter):
            return param
        return type(self)(param)

    def __call__(self):
        return self

    def __repr__(self):
        return float.__repr__(self)

    def __getattr__(self, item):
        raise AttributeError


class ConstantParameter(Parameter):
    def with_params(self, params):
        return self

    @property
    def params(self):
        # do not show up in the params
        return ()


class ParametrizedFunction(Parametrized):
    def __init__(self, f: Callable, *args, **params: Parametrized):
        self.f = partial(f, *args)
        super(ParametrizedFunction, self).__init__(*params.values())
        self.params_names = tuple(params.keys())
        self._mul = 1

    def __call__(self):
        return self._mul * self.f(**self.param_dict)

    def __getattr__(self, param):
        try:
            return super(ParametrizedFunction, self).__getattr__(param)
        except AttributeError:
            raise AttributeError(f"No parameter {param} in {self.f.func.__qualname__}")

    def __repr__(self):
        args = [f"{a}={v!r}" for a, v in zip(self.params_names, self._params)]
        return f"{self.f.func.__qualname__}({', '.join(args)})"

    def with_params(self, params):
        params = iter(params)
        new_params = []
        for param in self._params:
            new_params.append(param.with_params(params))
        return type(self)(self.f, **dict(zip(self.params_names, new_params)))
