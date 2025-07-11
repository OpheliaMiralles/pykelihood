from typing import Optional, TypeVar

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


def flatten_dict(d: dict):
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
    for k, v in d.items():
        if not isinstance(v, dict):
            res_dict[to_tuple(k)] = v
        else:
            new_dict = flatten_dict(v)
            for k_, v_ in new_dict.items():
                res_dict[to_tuple(k) + to_tuple(k_)] = v_
    return res_dict
