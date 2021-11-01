import torch.nn as nn

from . import *


conv = vars()


def conv_func(name: str, **kwargs) -> nn.Module:
    """
    Initialize the conv function. Can use all the torch.nn
    conv functions + WSConv2d and DepthWiseSeparableConv

    Args:
        name (str):
            The name of the conv function. Use lowercase letters.

    """
    assert name in conv["CONV_LOOKUP"].keys(), (
        f"Illegal conv func given. Allowed ones: {list(conv['CONV_LOOKUP'].keys())}"
    )

    kwargs = kwargs.copy()
    key = conv["CONV_LOOKUP"][name]
    conv_func = conv[key](**kwargs)

    return conv_func