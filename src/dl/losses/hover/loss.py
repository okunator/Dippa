import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_hv(tensor: torch.Tensor, kernel_size: int = 5, direction: str = "x"):
    """
    Computes first order derviatives in x or y direction using same sobel kernel
    as in the HoVer-net paper.   

    Args:
        tensor (torch.Tensor): input tensor. Shape (B, 1, H, W) or (B, H, W)
        kernel_size (int): size of the convolution kernel
        direction (str): direction of the derivative. One of ("x", "y")

    Returns:
        torch.Tensor computed 1st order derivatives of the input tensor. Shape (B, 2, H, W)
    """

    # Add channel dimension if shape (B, H, W)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    assert tensor.shape[1] == 1, f"Input tensor shape expected to have shape (B, H, W) or (B, 1, H, W). Got: {tensor.shape}" 
    assert kernel_size % 2 == 1, f"size must be odd. size: {kernel_size}"
    assert direction in ("x", "y"), "direction need to be one of ('x', 'y')"

    # Generate the sobel kernels
    range_h = torch.arange(-kernel_size//2+1, kernel_size//2+1, dtype=torch.float32, device=tensor.device)
    range_v = torch.arange(-kernel_size//2+1, kernel_size//2+1, dtype=torch.float32, device=tensor.device)
    h, v = torch.meshgrid(range_h, range_v)

    if direction == "x":
        kernel = h / (h*h + v*v + 1e-7)
        kernel = kernel.flip(0).unsqueeze(0).unsqueeze(0)
    elif direction == "y":
        kernel = v / (h*h + v*v + 1e-7)
        kernel = kernel.flip(1).unsqueeze(0).unsqueeze(0)

    # "SAME" padding to avoid losing height and width
    pad = [
        kernel.size(2) // 2,
        kernel.size(2) // 2,
        kernel.size(3) // 2,
        kernel.size(3) // 2
    ]
    pad_tensor = F.pad(tensor, pad, "replicate")

    # Compute the gradient
    grad = F.conv2d(pad_tensor, kernel)
    return grad


class HoVerLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Computes the loss for Horizontal and vertical branch from HoVer-Net.
        See: https://arxiv.org/abs/1812.06499
        """
        super(HoVerLoss, self).__init__()

    def forward(self,
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_inst: torch.Tensor,
                eps: float = 1e-7,
                **kwargs) -> torch.Tensor:
        """
        Computes the HoVer loss. I.e. mse for regressed HoVer maps against GT HoVer maps and
        gradient mse for the same inputs where 1st order sobel derivative is computed on the inputs

        Args:
            yhat (torch.Tensor): input tensor of size (B, 2, H, W). Regressed HoVer map 
            target (torch.Tensor): target tensor of shape (B, 2, H, W). Contains GT HoVer-maps 
            target_inst (torch.Tensor): target for instance segmentation used to focus loss to the
                                        correct nucleis. Shape (B, H, W)

        Returns:
            torch.Tensor: computed HoVer loss (scalar)
        """
        # Compute mse loss
        loss_mse = yhat - target
        loss_mse = (loss_mse*loss_mse).mean()

        # Compute msge loss
        pred_grad_x = sobel_hv(yhat[:, 0, ...], direction="x")
        pred_grad_y = sobel_hv(yhat[:, 1, ...], direction="y")
        pred_grad = torch.stack([pred_grad_x.squeeze(1), pred_grad_y.squeeze(1)], dim=1)

        target_grad_x = sobel_hv(target[:, 0, ...], direction="x")
        target_grad_y = sobel_hv(target[:, 1, ...], direction="y")
        target_grad = torch.stack([target_grad_x.squeeze(1), target_grad_y.squeeze(1)], dim=1)

        focus = torch.stack([target_inst, target_inst], dim=1)
        loss_msge = pred_grad - target_grad
        loss_msge = focus*(loss_msge * loss_msge)
        loss_msge = loss_msge.sum() / focus.clamp_min(eps).sum()

        # Compute the total loss
        loss = loss_msge + loss_mse 
        return loss
