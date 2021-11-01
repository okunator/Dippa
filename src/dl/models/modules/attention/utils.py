import torch.nn as nn

from . import *


att = vars()


def att_func(name: str=None, **kwargs) -> nn.Module:
    """
    Initialize the attention function.

    Args:
        name (str, default=None):
            The name of the attention function. Use lowercase letters.

    """
    allowed = [*att['ATT_LOOKUP'].keys(), None]

    assert name in allowed, (
        f"Illegal att func given. Allowed ones: {allowed}"
    )

    if name is not None:
        kwargs = kwargs.copy()
        key = att["ATT_LOOKUP"][name]
        att_f = att[key](**kwargs)
    else:
        att_f = nn.Identity()

    return att_f