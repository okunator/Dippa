import torch
import torch.nn as nn
import torch.nn.functional as F



@torch.jit.script
def mish_jit_fwd(x: torch.Tensor) -> torch.Tensor:
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()

    return grad_output.mul(
        x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp)
    )


class MishFunction(torch.autograd.Function):
    """
    Memory efficient mish function implementation

    Mish: A Self Regularized Non-Monotonic Neural Activation Function:
    https://arxiv.org/abs/1908.08681
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
    Apply element-wise mish-activation

    https://github.com/digantamisra98/Mish
    """
    return MishFunction.apply(x)


class Mish(nn.Module):
    """
    Element-wise mish
    
    https://github.com/digantamisra98/Mish
    """

    def __init__(self, inplace: bool=False) -> None:
        """
        Args:
        -------
            inplace (bool, default=False): 
                This is not used, exists only for compatibility
        """
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.

        Args:
        --------
            input (torch.Tensor):
                input tensor. Can be of any shape (C, *)

        Returns:
        --------
            torch.Tensor: activated output tensor. Shape same as input.
        """
        return mish(input)