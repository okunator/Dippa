import torch
import torch.nn as nn
from typing import Optional

from . import *


loss = vars()


def loss_func(
        name: str,
        edge_weight: Optional[float]=None,
        class_weights: Optional[torch.Tensor]=None,
        **kwargs
    ) -> nn.Module:
    """
    Initialize the loss function.

    Args:
    -----------
        name (str):
            The name of the loss function. Use lowercase letters.
        edge_weight (float, optional, default=none): 
            Weight to be added to nuclei borders like in Unet paper
        class_weights (torch.Tensor, optional, default=None): 
            Optional tensor of size (n_classes,) for class weights

    Returns:
    -----------
        nn.Module: Initialized nn.Module.
    """
    allowed = loss['SEG_LOSS_LOOKUP'].keys()
    assert name in allowed, (
        f"Illegal loss func given. Allowed ones: {allowed}"
    )

    key = loss["SEG_LOSS_LOOKUP"][name]
    loss_f = loss[key](
        edge_weight=edge_weight, class_weights=class_weights, **kwargs
    )

    return loss_f


__all__ = ["loss_func"]