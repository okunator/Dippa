import torch.nn as nn

from . import *


merges = vars()


def merge_func(name: str, **kwargs) -> nn.Module:
    """
    Initialize the merging function.

    Args:
    -----------
        name (str):
            The name of the merging type. Use lowercase letters.
            One of "concatenate", "summation"

    Returns:
    -----------
        nn.Module: Initialized nn.Module.
    """
    allowed_merges = merges['MERGE_LOOKUP'].keys()
    assert name in allowed_merges, (
        f"Illegal merge given. Allowed ones: {allowed_merges}. Got {name}"
    )

    kwargs = kwargs.copy()
    key = merges["MERGE_LOOKUP"][name]
    merge_f = merges[key](**kwargs)

    return merge_f