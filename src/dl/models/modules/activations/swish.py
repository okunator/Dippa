import torch
import torch.nn as nn


@torch.jit.script
def swish_jit_fwd(input: torch.Tensor) -> torch.Tensor:
    return input.mul(torch.sigmoid(input))


@torch.jit.script
def swish_jit_bwd(input: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    input_s = torch.sigmoid(input)
    return grad_output * (input_s * (1 + input * (1 - input_s)))


class SwishFunction(torch.autograd.Function):
    """
    Memory efficient Swish implementation.
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
    Applies the element-wise swish function
    """

    def __init__(self, inplace: bool = False) -> None:
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
        return swish(input)