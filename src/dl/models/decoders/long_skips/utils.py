import torch.nn as nn

from . import *


long_skips = vars()


def long_skip_func(name: str, **kwargs) -> nn.Module:
    """
    Initialize the long skip function.

    Args:
    -----------
        name (str):
            The name of the long skip type. Use lowercase letters.
            One of "unet", "unet3+", "unet++"

    Returns:
    -----------
        nn.Module: Initialized nn.Module.

    """
    allowed_skips = [*long_skips['SKIP_LOOKUP'].keys(), None]
    assert name in allowed_skips, (
        f"Illegal long skip given. Allowed ones: {allowed_skips}. Got {name}"
    )

    kwargs = kwargs.copy()
    key = long_skips["SKIP_LOOKUP"][name]
    skip_f = long_skips[key](**kwargs)

    return skip_f


__all__ = ["long_skip_func"]