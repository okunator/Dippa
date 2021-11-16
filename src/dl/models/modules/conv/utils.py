import torch.nn as nn

from . import *


convs = vars()


def conv_block_func(name: str, skip_type: str, **kwargs) -> nn.Module:
    """
    Initialize the convolution block function.

    Args:
    -----------
        name (str):
            The name of the conv block type. Use lowercase letters.
            One of "basic", "bottleneck", "mbconv", "fusedmbconv", "dws"
        skip_type (str):
            One of "basic", "residual", "dense"

    Returns:
    -----------
        nn.Module: Initialized nn.Module.

    """
    allowed_skips = convs['CONV_LOOKUP'].keys()
    assert skip_type in allowed_skips, (
        f"Illegal skip type given. Allowed ones: {allowed_skips}. Got: {skip_type}"
    )

    allowed_convs = convs['CONV_LOOKUP'][skip_type].keys()
    assert name in allowed_convs, (
        f"Illegal conv type given. Allowed ones: {allowed_convs}. Got: {name}"
    )

    kwargs = kwargs.copy()
    key = convs["CONV_LOOKUP"][skip_type][name]
    conv_f = convs[key](**kwargs)

    return conv_f


__all__ = ["conv_block_func"]
