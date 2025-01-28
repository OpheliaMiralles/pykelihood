from collections.abc import Iterable, MutableSequence
from typing import Dict, Optional, TypeVar

T = TypeVar("T")


def ifnone(x: Optional[T], default: T) -> T:
    """
    Return `default` if `x` is None, otherwise return `x`.

    Parameters
    ----------
    x : Optional[T]
        The value to check.
    default : T
        The default value to return if `x` is None.

    Returns
    -------
    T
        `x` if it is not None, otherwise `default`.
    """
    return default if x is None else x


def to_tuple(x):
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def flatten_dict(dict: Dict):
    """
    Flatten a nested dictionary.

    Parameters
    ----------
    d : Dict[str, Any]
        The dictionary to flatten.
    parent_key : str, optional
        The base key to use for the flattened keys, by default "".
    sep : str, optional
        The separator to use between keys, by default "_".

    Returns
    -------
    Dict[str, Any]
        The flattened dictionary.
    """
    res_dict = {}
    for k, v in dict.items():
        if not isinstance(v, Dict):
            res_dict[to_tuple(k)] = v
        else:
            new_dict = flatten_dict(v)
            for k_, v_ in new_dict.items():
                res_dict[to_tuple(k) + to_tuple(k_)] = v_
    return res_dict


def hash_with_series(*args, **kwargs):
    to_hash = []
    try:
        import pandas as pd
    except ImportError:
        pd = None
    for v in args:
        if pd is not None and isinstance(v, pd.Series):
            v = tuple(v.values)
        elif isinstance(v, MutableSequence) or isinstance(v, Iterable):
            v = tuple(v)
        to_hash.append(v)
    for k, v in kwargs.items():
        v = hash_with_series(v)
        to_hash.append((k, v))
    return hash(tuple(to_hash))
