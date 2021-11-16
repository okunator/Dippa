import torch.nn as nn

from . import *


up = vars()


def up_func(name: str, **kwargs) -> nn.Module:
    """
    Initialize the upsampling function. Can use torch.nn
    upsampling functions plus fixed unpooling

    Args:
    -----------
        name (str):
            The name of the upsampling function. Use lowercase letters.
    
    Returns:
    -----------
        nn.Module: Initialized nn.Module.
    """

    allowed = up['UP_LOOKUP'].keys()

    assert name in allowed, (
        f"Illegal upsampling func given. Allowed ones: {allowed}"
    )

    kwargs = kwargs.copy()
    key = up["UP_LOOKUP"][name]
    up_f = up[key](**kwargs)

    return up_f


__all__ = ["up_func"]
