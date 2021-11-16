import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.utils import sobel_hv, filter2D


def grad_mse(yhat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient mse loss  

    Args:
    ---------
        yhat (torch.Tensor): 
            Input tensor of size (B, 2, H, W). Regressed HoVer map 
        target (torch.Tensor): 
            Target tensor of shape (B, 2, H, W). Contains GT HoVer-maps

    Returns:
    ----------
        torch.Tensor: Computed gradient mse loss matrix. Shape (B, H, W)

    """
    kernel = sobel_hv(window_size = 5, device=yhat.device)
    grad_yhat = filter2D(yhat, kernel)
    grad_target = filter2D(target, kernel)
    msge = F.mse_loss(grad_yhat, grad_target, reduction="none")

    return msge


class GradMSE(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Computes the gradient MSE loss for horizontal and vertical
        branch from HoVer-Net. See: https://arxiv.org/abs/1812.06499
        """
        super().__init__()
        self.eps = 1e-6

    def forward(
            self,
            yhat: torch.Tensor,
            target: torch.Tensor,
            target_inst: torch.Tensor=None,
            **kwargs
        ) -> torch.Tensor:
        """
        Computes msge-loss

        Args:
        ----------
            yhat (torch.Tensor): 
                Input tensor (B, 2, H, W). Regressed HoVer maps
            target (torch.Tensor): 
                Target tensor (B, 2, H, W). Contains GT HoVer maps 
            target_inst (torch.Tensor): 
                instance map target that  used to focus loss to the 
                correct nuclei. Shape (B, H, W).

        Returns:
        ----------
            torch.Tensor. Computed gradient mse loss (scalar)
        """
        focus = torch.stack([target_inst, target_inst], dim=1)
        loss = focus*grad_mse(yhat, target)
        loss = loss.sum() / focus.clamp_min(self.eps).sum()

        return loss
