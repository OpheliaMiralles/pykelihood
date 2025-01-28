import numpy as np
from typing import Any, Dict, Optional, Tuple, TypeVar

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


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
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
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def count() -> int:
    """
    Generate an infinite sequence of integers starting from 0.

    Yields
    ------
    int
        The next integer in the sequence.
    """
    i = 0
    while True:
        yield i
        i += 1
