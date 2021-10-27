import torch.nn as nn

from . import *


norm = vars()


def norm_func(name: str, **kwargs) -> nn.Module:
    """
    Initialize the activation function. Can use all the torch.nn
    activation functions and Swish and Mish 

    Args:
        name (str):
            The name of the activation function. Use lowercase letters.

    """
    allowed = [*norm['NORM_LOOKUP'].keys(), None]

    assert name in allowed, (
        f"Illegal norm func given. Allowed ones: {allowed}"
    )

    if name is not None:
        kwargs = kwargs.copy()
        key = norm["NORM_LOOKUP"][name]
        norm_func = norm[key](**kwargs)
    else:
        norm_func = nn.Identity()

    return norm_func
