from typing import Dict


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
