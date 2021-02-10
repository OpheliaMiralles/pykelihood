from typing import Dict, Iterable, MutableSequence


def ifnone(x, default):
    if x is None:
        return default
    return x


def to_tuple(x):
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def flatten_dict(dict: Dict):
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
