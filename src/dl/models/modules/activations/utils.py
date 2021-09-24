import torch.nn as nn

from .mish import Mish
from .swish import Swish


def convert_relu_to_mish(model: nn.Module) -> None:
    """
    Convert ReLU activations in a give model to Mish

    Args:
    --------
        model (nn.Module):
            pytorch model specification
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish(inplace=False))
        else:
            convert_relu_to_mish(child)


def convert_relu_to_swish(model: nn.Module) -> None:
    """
    Convert ReLU activations in a give model to Mish

    Args:
    ----------
        model (nn.Module):
            pytorch model specification
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Swish(inplace=False))
        else:
            convert_relu_to_swish(child)