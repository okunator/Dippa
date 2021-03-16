import torch
import torch.nn.functional as F

def filter2D(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolves a given kernel on input tensor without losing dimensional shape

    Args:
    ----------
        input_tensor (torch.Tensor): 
            Input image/tensor
        kernel (torch.Tensor):
            Convolution kernel/window

    Returns:
    ----------
        Convolved torch.Tensor. Same shape as input
    """

    (_, channel, _, _) = input_tensor.size()

    # "SAME" padding to avoid losing height and width
    pad = [
        kernel.size(2) // 2,
        kernel.size(2) // 2,
        kernel.size(3) // 2,
        kernel.size(3) // 2
    ]
    pad_tensor = F.pad(input_tensor, pad, "replicate")

    out = F.conv2d(pad_tensor, kernel, groups=channel)
    return out