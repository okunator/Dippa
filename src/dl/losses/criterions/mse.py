import torch
import torch.nn as nn
import torch.nn.functional as F


# Quick MSE wrapper

class MSE(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs):
        """
        Quick wrapper class for torch MSE

        Args:
        --------
            reduction (str, default="mean"):
                reduction method for the computed loss matrix
        """
        super(MSE, self).__init__()
        self.reduction = reduction

    def forward(self, yhat: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
        ---------
            yhat (torch.Tensor): 
                Input tensor of size (B, C, H, W)
            target (torch.Tensor): 
                Target tensor of size (B, H, W), where values 
                of a vector correspond to class index
        """
        return F.mse_loss(yhat, target, reduction=self.reduction)
