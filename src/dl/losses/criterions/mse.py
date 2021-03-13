import torch
import torch.nn as nn
import torch.nn.functional as F


# Quick MSE wrapper

class MSE(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs):
        super(MSE, self).__init__()
        self.reduction = reduction

    def forward(self, yhat: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.mse_loss(yhat, target, reduction=self.reduction)
