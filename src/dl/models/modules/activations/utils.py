import torch.nn as nn

from . import *


act = vars()


def act_func(name: str=None, **kwargs) -> nn.Module:
    """
    Initialize the activation function. Can use all the torch.nn
    activation functions and Swish and Mish 

    Args:
        name (str, default=None):
            The name of the activation function. Use lowercase letters.

    """
    allowed = [*act['ACT_LOOKUP'].keys(), None]

    assert name in allowed, (
        f"Illegal act func given. Allowed ones: {allowed}"
    )

    if name is not None:
        kwargs = kwargs.copy()
        kwargs["inplace"] = True
        key = act["ACT_LOOKUP"][name]
        act_f = act[key](**kwargs)
    else:
        act_f = nn.Identity()

    return act_f
