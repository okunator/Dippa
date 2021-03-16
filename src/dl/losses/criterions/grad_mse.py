import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dl.filters.kernels import sobel_hv
from src.dl.filters.filter import filter2D


def grad_mse(yhat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradients of yhat and target to compute mse(grad_yhat, grad_target)

    Args:
    ---------
        yhat (torch.Tensor): 
            Input tensor of size (B, 2, H, W). Regressed HoVer map 
        target (torch.Tensor): 
            Target tensor of shape (B, 2, H, W). Contains GT HoVer-maps 
    """
    kernel = sobel_hv(window_size = 5, device=yhat.device)
    grad_yhat = filter2D(yhat, kernel)
    grad_target = filter2D(target, kernel)
    msge = F.mse_loss(grad_yhat, grad_target, reduction="none")
    return msge


class GradMSE(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Computes the gradient MSE loss for horizontal and vertical branch from HoVer-Net.
        See: https://arxiv.org/abs/1812.06499
        """
        super(GradMSE, self).__init__()
        self.eps = 1e-6

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_inst: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Computes the gradients of regressed HoVer and GT HoVer maps
        that are used as yhat and target in mse loss

        Args:
        ----------
            yhat (torch.Tensor): 
                Input tensor of size (B, 2, H, W). Regressed HoVer map 
            target (torch.Tensor): 
                Target tensor of shape (B, 2, H, W). Contains GT HoVer-maps 
            target_inst (torch.Tensor): 
                Target for instance segmentation used to focus loss to the
                correct nucleis. Shape (B, H, W)
        """
        focus = torch.stack([target_inst, target_inst], dim=1)
        loss = focus*grad_mse(yhat, target)
        loss = loss.sum() / focus.clamp_min(self.eps).sum()
        return loss
