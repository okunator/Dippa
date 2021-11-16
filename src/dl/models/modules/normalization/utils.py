import torch.nn as nn

from . import *


norm = vars()


def norm_func(name: str=None, **kwargs) -> nn.Module:
    """
    Initialize the normalization function. Can use all the torch.nn
    normalization functions plus batch channel norm

    Args:
    -----------
        name (str, default=None):
            The name of the norm function. Use lowercase letters.

    Returns:
    -----------
        nn.Module: Initialized nn.Module.

    """
    allowed = [*norm['NORM_LOOKUP'].keys(), None]

    assert name in allowed, (
        f"Illegal norm func given. Allowed ones: {allowed}"
    )

    if name is not None:
        kwargs = kwargs.copy()
        key = norm["NORM_LOOKUP"][name]
        norm_f = norm[key](**kwargs)
    else:
        norm_f = nn.Identity()

    return norm_f


__all__ = ["norm_func"]
