import torch
import torch.nn as nn
import torch.nn.functional as F


# Ported from: https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/modules/activations.py

@torch.jit.script
def mish_jit_fwd(x: torch.Tensor) -> torch.Tensor:
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishFunction(torch.autograd.Function):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish(x: torch.Tensor):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return MishFunction.apply(x)


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    Credit: https://github.com/digantamisra98/Mish
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
        return mish(input)