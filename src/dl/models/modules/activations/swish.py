import torch
import torch.nn as nn
import torch.nn.functional as F

# Ported from: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/modules/activations.py

@torch.jit.script
def swish_jit_fwd(x: torch.Tensor) -> torch.Tensor:
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def swish_jit_bwd(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishFunction(torch.autograd.Function):
    """
    Memory efficient Swish implementation.
    Credit:
        https://blog.ceshine.net/post/pytorch-memory-swish/
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/activations_jit.py
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish(x: torch.Tensor) -> torch.Tensor:
    return SwishFunction.apply(x)


class Swish(nn.Module):
    """
    Applies the swish function element-wise:

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    
    Examples:
        >>> s = Swish()
        >>> input = torch.randn(2)
        >>> output = s(input)
    """

    def __init__(self, inplace: bool = False) -> None:
        """
        Init method.
        :param inplace: Not used, exists only for compatibility
        """
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.
        """
        return swish(input)