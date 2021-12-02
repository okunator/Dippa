import torch.nn as nn

from . import *


conv = vars()


def conv_func(name: str, **kwargs) -> nn.Module:
    """
    Initialize the conv function. Can use all the torch.nn
    conv functions + WSConv2d and DepthWiseSeparableConv

    Args:
    -----------
        name (str):
            The name of the conv function. Use lowercase letters.

    Returns:
    -----------
        nn.Module: Initialized nn.Module.
    """
    allowed = list(conv["CONV_LOOKUP"].keys())
    assert name in allowed, (
        f"Illegal conv func given. Got: {name}. Allowed: {allowed}"
    )

    kwargs = kwargs.copy()
    key = conv["CONV_LOOKUP"][name]
    conv_func = conv[key](**kwargs)

    return conv_func


__all__ = ["conv_func"]