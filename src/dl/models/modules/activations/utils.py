import torch.nn as nn

from . import *


act = vars()


def act_func(name: str, **kwargs) -> nn.Module:
    """
    Initialize the activation function. Can use all the torch.nn
    activation functions and Swish and Mish 

    Args:
        name (str):
            The name of the activation function. Use lowercase letters.

    """
    assert name in act["ACT_LOOKUP"].keys(), (
        f"Illegal act func given. Allowed ones: {list(act['ACT_LOOKUP'].keys())}"
    )

    kwargs = kwargs.copy()
    kwargs["inplace"] = True
    key = act["ACT_LOOKUP"][name]
    act_func = act[key](**kwargs)

    return act_func
