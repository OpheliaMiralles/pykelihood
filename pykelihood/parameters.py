from __future__ import annotations

from collections import ChainMap
from typing import Any, Callable, Dict, Iterable, Tuple, TypeVar, Union

from pykelihood.utils import flatten_dict

_T = TypeVar("_T")


def ensure_parametrized(x: Any, constant=False) -> Parametrized:
    if isinstance(x, Parametrized):
        return x
    cls = ConstantParameter if constant else Parameter
    return cls(x)


class Parametrized(object):
    params_names: Tuple[str]

    def __init__(self, *params: Union[Parametrized, Any]):
        self._params = tuple(ensure_parametrized(p) for p in params)

    def _build_instance(self, **new_params):
        return type(self)(**new_params)

    @property
    def params(self) -> Tuple[Parametrized]:
        return self._params

    @property
    def param_dict(self) -> Dict[str, Parametrized]:
        return dict(zip(self.params_names, self.params))

    @property
    def flattened_params(self) -> Tuple[Parametrized]:
        return tuple(p_ for p in self.params for p_ in p.params)

    @property
    def flattened_param_dict(self) -> Dict[str, Parametrized]:
        p_dict = flatten_dict(self._flattened_param_dict_helper())
        return {"_".join(names): value for names, value in p_dict.items()}

    def _flattened_param_dict_helper(self):
        return {
            name: value._flattened_param_dict_helper()
            for name, value in self.param_dict.items()
        }

    def param_mapping(self, only_opt=False):
        results: "list[list[Parametrized, list[str]]]" = []
        unique = []
        for q, param_name in zip(
            (p_ for p in self.flattened_params for p_ in p.params),
            self.flattened_param_dict,
        ):
            if not (only_opt and isinstance(q, ConstantParameter)):
                if q not in unique:
                    unique.append(q)
                    results.append([q, [param_name]])
                else:
                    for q_, names in results:
                        if q_ is q:
                            names.append(param_name)
        return results

    @property
    def optimisation_params(self) -> Tuple[Parametrized]:
        unique = []
        for q in (p_ for p in self.params for p_ in p.optimisation_params):
            if q not in unique:
                unique.append(q)
        return unique

    @property
    def optimisation_param_dict(self) -> Dict[str, Parametrized]:
        p_dict = flatten_dict(self._optimisation_param_dict_helper())
        return {"_".join(names): value for names, value in p_dict.items()}

    def _optimisation_param_dict_helper(self):
        return {
            name: value._optimisation_param_dict_helper()
            for name, value in self.param_dict.items()
            if not isinstance(value, ConstantParameter)
        }

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("A generic Parametrized object has no value!")

    def __repr__(self):
        args = [f"{a}={v!r}" for a, v in zip(self.params_names, self._params)]
        return f"{type(self).__name__}({', '.join(args)})"

    def with_params(self, params: Iterable = None, **named_params):
        if params is not None and named_params:
            raise ValueError("Please only use one way to provide values to parameters.")
        if params is not None:
            params = iter(params)
            mapping = self.param_mapping(only_opt=True)
            new_params = {}
            for (param, names), new_param in zip(mapping, params):
                new_param_obj = param.with_params([new_param])
                for name in names:
                    new_params[name] = new_param_obj
            return self.with_params(**new_params)
            # for p_name, param in self.param_dict.items():
            #     new_params[p_name] = param.with_params(params)
        else:
            new_params = {}
            for p_name, param in self.param_dict.items():
                this_param_values = {}
                for name, value in named_params.items():
                    # check if this value is for this parameter
                    if name == p_name:
                        new_params[name] = value
                        # no need to look further
                        break
                    elif name.startswith(p_name + "_"):
                        remaining = name[len(p_name + "_") :]
                        this_param_values[remaining] = value
                else:  # nobreak
                    if this_param_values:
                        new_params[p_name] = param.with_params(**this_param_values)
            # use current values if none were given
            new_params = ChainMap(new_params, self.param_dict)
        return self._build_instance(**new_params)

    def __getattr__(self, param: str) -> Parametrized:
        try:
            idx = self.params_names.index(param)
        except ValueError:
            raise AttributeError(f"No parameter {param} in {type(self).__name__}")
        return self._params[idx]


class Parameter(Parametrized):
    def __init__(self, value=0.0):
        self._value = value

    @property
    def params(self):
        return (self,)

    def _flattened_param_dict_helper(self):
        return self

    @property
    def optimisation_params(self):
        return (self,)

    def _optimisation_param_dict_helper(self):
        return self

    def with_params(self, params):
        param = next(iter(params))
        if isinstance(param, ConstantParameter):
            return param
        return type(self)(param)

    @property
    def value(self):
        return self._value

    def __call__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)

    def __getattr__(self, item):
        raise AttributeError

    def __float__(self):
        return float(self.value)


class ConstantParameter(Parameter):
    def with_params(self, params):
        return self

    @property
    def optimisation_params(self):
        # do not show up in the params
        return ()


class ParametrizedFunction(Parametrized):
    def __init__(self, f: Callable, *, fname=None, **params: Parametrized):
        super(ParametrizedFunction, self).__init__(*params.values())
        self.params_names = tuple(params.keys())
        self.f = f
        self.fname = fname or f.__qualname__

    def __call__(self, *args, **kwargs):
        param_values = {p_name: p() for p_name, p in self.param_dict.items()}
        return self.f(*args, **kwargs, **param_values)

    def __getattr__(self, param):
        try:
            return super(ParametrizedFunction, self).__getattr__(param)
        except AttributeError:
            raise AttributeError(f"No parameter {param} in {self.fname}")

    def __repr__(self):
        args = [f"{a}={v!r}" for a, v in zip(self.params_names, self._params)]
        return f"{self.fname}({', '.join(args)})"

    def _build_instance(self, **new_params):
        return type(self)(self.f, fname=self.fname, **new_params)
