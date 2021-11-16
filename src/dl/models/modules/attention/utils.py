import torch.nn as nn

from . import *


att = vars()


def att_func(name: str=None, **kwargs) -> nn.Module:
    """
    Initialize the attention function.

    Args:
    -----------
        name (str, default=None):
            The name of the attention function. Use lowercase letters.

    Returns:
    -----------
        nn.Module: Initialized nn.Module.
    """
    allowed = [*att['ATT_LOOKUP'].keys(), None]
    assert name in allowed, (
        f"Illegal att func given. Allowed ones: {allowed}"
    )
    
    if name is not None and kwargs["in_channels"] > 2:
        key = att["ATT_LOOKUP"][name]
        att_f = att[key](**kwargs)
    else:
        att_f = nn.Identity()

    return att_f


__all__ = ["att_func"]