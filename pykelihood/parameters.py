from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
import numpy.typing as npt

from pykelihood.utils import flatten_dict

if TYPE_CHECKING:
    from typing import Self

_T = TypeVar("_T")


def ensure_parametrized(x: Any, constant=False) -> Parametrized:
    """
    Ensure the input is a `Parametrized` object.

    Parameters
    ----------
    x : Any
        Input value to be converted to `Parametrized`.
    constant : bool, optional
        If True, convert to `ConstantParameter`, by default False.

    Returns
    -------
    Parametrized
        The `Parametrized` object.
    """
    if isinstance(x, Parametrized):
        return x
    cls = ConstantParameter if constant else Parameter
    return cls(x)


class Parametrized:
    """
    Base class for parametrized objects.
    """

    params_names: tuple[str, ...]

    def __init__(self, *params: Parametrized | Any):
        """
        Initialize the `Parametrized` object.

        Parameters
        ----------
        params : Union[Parametrized, Any]
            Parameters to be used in the object.
        """
        self._params = tuple(ensure_parametrized(p) for p in params)

    def _build_instance(self, **new_params) -> Self:
        """
        Build a new instance with the given parameters.

        Parameters
        ----------
        new_params : dict
            New parameters for the instance.

        Returns
        -------
        Parametrized
            The new instance.
        """
        return type(self)(**new_params)

    @property
    def params(self) -> tuple[Parametrized]:
        """
        Get parameters in their parametrized format, e.g. how they were defined.

        Returns
        -------
        Tuple[Parametrized]
            The parameters.
        """
        return self._params

    @property
    def param_dict(self) -> dict[str, Parametrized]:
        """
        Get a dictionary of parameter names and their values.

        Returns
        -------
        Dict[str, Parametrized]
            The parameter dictionary.
        """
        return dict(zip(self.params_names, self.params))

    @property
    def flattened_params(self) -> tuple[Parametrized]:
        """
        Get a horizontal view of all parameters in the final state of their
        respective tree of dependence.

        Returns
        -------
        Tuple[Parametrized]
            The flattened parameters.
        """
        return tuple(p_ for p in self.params for p_ in p.params)

    @property
    def flattened_param_dict(self) -> dict[str, Parametrized]:
        """
        Get a dictionary of flattened parameter names and their values.

        Returns
        -------
        Dict[str, Parametrized]
            The flattened parameter dictionary.
        """
        p_dict = flatten_dict(self._flattened_param_dict_helper())
        return {"_".join(names): value for names, value in p_dict.items()}

    def _flattened_param_dict_helper(self):
        """
        Helper function to flatten the parameter dictionary.

        Returns
        -------
        dict
            The flattened parameter dictionary.
        """
        return {
            name: value._flattened_param_dict_helper()
            for name, value in self.param_dict.items()
        }

    def param_mapping(self, only_opt=False):
        """
        Map parameters to their names.

        Parameters
        ----------
        only_opt : bool, optional
            If True, only include optimization parameters, by default False.

        Returns
        -------
        list
            The parameter mapping.
        """
        results: list[list[Parametrized, list[str]]] = []
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
    def optimisation_params(self) -> tuple[Parametrized]:
        """
        Get all parameters used in the optimization.

        Returns
        -------
        Tuple[Parametrized]
            The optimization parameters.
        """
        unique = []
        for q in (p_ for p in self.params for p_ in p.optimisation_params):
            if q not in unique:
                unique.append(q)
        return unique

    @property
    def optimisation_param_dict(self) -> dict[str, Parametrized]:
        """
        Get a dictionary of optimization parameter names and their values.

        Returns
        -------
        Dict[str, Parametrized]
            The optimization parameter dictionary.
        """
        p_dict = flatten_dict(self._optimisation_param_dict_helper())
        return {"_".join(names): value for names, value in p_dict.items()}

    def _optimisation_param_dict_helper(self):
        """
        Helper function to get the optimization parameter dictionary.

        Returns
        -------
        dict
            The optimization parameter dictionary.
        """
        return {
            name: value._optimisation_param_dict_helper()
            for name, value in self.param_dict.items()
            if not isinstance(value, ConstantParameter)
        }

    def __call__(self, *args, **kwargs):
        """
        Call the `Parametrized` object.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("A generic Parametrized object has no value!")

    def __repr__(self):
        """
        Get the string representation of the `Parametrized` object.

        Returns
        -------
        str
            The string representation.
        """
        args = [f"{a}={v!r}" for a, v in zip(self.params_names, self._params)]
        return f"{type(self).__name__}({', '.join(args)})"

    def with_params(self, params: Iterable | None = None, **named_params) -> Self:
        """
        Create a new instance of the object with the given parameters.

        Parameters
        ----------
        params : Iterable, optional
            Sequence of parameters in the same format as the `params` property, by default None.
        named_params : dict, optional
            Optionally, provide the parameters by name.

        Returns
        -------
        Parametrized
            The new instance.

        Raises
        ------
        ValueError
            If both `params` and `named_params` are provided.
        """
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
        """
        Get the value of a parameter by name.

        Parameters
        ----------
        param : str
            Name of the parameter.

        Returns
        -------
        Parametrized
            The parameter value.

        Raises
        ------
        AttributeError
            If the parameter is not found.
        """
        try:
            idx = self.params_names.index(param)
        except ValueError:
            raise AttributeError(f"No parameter {param} in {type(self).__name__}")
        return self._params[idx]


class Parameter(Parametrized):
    """
    Class for a single parameter.
    """

    def __init__(self, value: npt.ArrayLike) -> None:
        """
        Initialize the `Parameter` object.

        Parameters
        ----------
        value : float or ndarray
            Initial value of the parameter
        """
        self._value = np.asarray(value, dtype=np.float64)

    @property
    def params(self):
        """
        Get the parameter itself.

        Returns
        -------
        Tuple[Parametrized]
            The parameter.
        """
        return (self,)

    def _flattened_param_dict_helper(self):
        """
        Helper function to flatten the parameter dictionary.

        Returns
        -------
        Parameter
            The parameter itself.
        """
        return self

    @property
    def optimisation_params(self):
        """
        Get the parameter itself for optimization.

        Returns
        -------
        Tuple[Parametrized]
            The parameter.
        """
        return (self,)

    def _optimisation_param_dict_helper(self):
        """
        Helper function to get the optimization parameter dictionary.

        Returns
        -------
        Parameter
            The parameter itself.
        """
        return self

    def with_params(self, params):
        """
        Create a new instance of the parameter with the given value.

        Parameters
        ----------
        params : Iterable
            New value for the parameter.

        Returns
        -------
        Parameter
            The new instance.
        """
        param = next(iter(params))
        if isinstance(param, ConstantParameter):
            return param
        return type(self)(param)

    @property
    def value(self) -> npt.NDArray[np.float64]:
        """
        Get the value of the parameter.

        Returns
        -------
        ndarray
            The value.
        """
        return self._value

    def __call__(self):
        """
        Call the parameter to get its value.

        Returns
        -------
        float
            The value.
        """
        return self.value

    def __repr__(self):
        """
        Get the string representation of the parameter.

        Returns
        -------
        str
            The string representation.
        """
        return repr(self.value)

    def __getattr__(self, item):
        """
        Raise an `AttributeError` for undefined attributes.

        Parameters
        ----------
        item : str
            Attribute name.

        Raises
        ------
        AttributeError
            Always raised.
        """
        raise AttributeError


class ConstantParameter(Parameter):
    """
    Utility class used to manage parameters that are not optimized.
    """

    def with_params(self, params):
        """
        Return the parameter itself, as it is constant.

        Parameters
        ----------
        params : Iterable
            Ignored.

        Returns
        -------
        ConstantParameter
            The parameter itself.
        """
        return self

    @property
    def optimisation_params(self):
        """
        Do not include in optimization parameters.

        Returns
        -------
        Tuple
            Empty tuple.
        """
        # do not show up in the params
        return ()


class ParametrizedFunction(Parametrized):
    """
    Class for a parametrized function.
    """

    def __init__(self, f: Callable, *, fname=None, **params: Parametrized):
        """
        Initialize the `ParametrizedFunction` object.

        Parameters
        ----------
        f : Callable
            The function to be parametrized.
        fname : str, optional
            Name of the function, by default None.
        params : Parametrized
            Parameters for the function.
        """
        super().__init__(*params.values())
        self.params_names = tuple(params.keys())
        self.f = f
        self.fname = fname or f.__qualname__

    def __call__(self, *args, **kwargs):
        """
        Call the parametrized function with the given arguments.

        Parameters
        ----------
        args : tuple
            Positional arguments for the function.
        kwargs : dict
            Keyword arguments for the function.

        Returns
        -------
        Any
            The result of the function call.
        """
        param_values = {p_name: p() for p_name, p in self.param_dict.items()}
        return self.f(*args, **kwargs, **param_values)

    def __getattr__(self, param):
        """
        Get the value of a parameter by name.

        Parameters
        ----------
        param : str
            Name of the parameter.

        Returns
        -------
        Parametrized
            The parameter value.

        Raises
        ------
        AttributeError
            If the parameter is not found.
        """
        try:
            return super().__getattr__(param)
        except AttributeError:
            raise AttributeError(f"No parameter {param} in {self.fname}")

    def __repr__(self):
        """
        Get the string representation of the parametrized function.

        Returns
        -------
        str
            The string representation.
        """
        args = [f"{a}={v!r}" for a, v in zip(self.params_names, self._params)]
        return f"{self.fname}({', '.join(args)})"

    def _build_instance(self, **new_params):
        """
        Build a new instance with the given parameters.

        Parameters
        ----------
        new_params : dict
            New parameters for the instance.

        Returns
        -------
        ParametrizedFunction
            The new instance.
        """
        return type(self)(self.f, fname=self.fname, **new_params)
