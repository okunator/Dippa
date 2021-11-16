import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.dl.utils import one_hot
from .._base._weighted_base_loss import WeightedBaseLoss


class MSE(WeightedBaseLoss):
    def __init__(
            self,
            edge_weight: Optional[float]=None,
            class_weights: Optional[torch.Tensor]=None,
            **kwargs
        ) -> None:
        """
        Quick wrapper class for torch MSE

        Args:
        --------
            reduction (str, default="mean"):
                reduction method for the computed loss matrix
        """
        super().__init__(class_weights, edge_weight)

    def forward(
            self,
            yhat: torch.Tensor,
            target: torch.Tensor,
            target_weight: Optional[torch.Tensor]=None,
            **kwargs
        ) -> torch.Tensor:
        """
        Args:
        ---------
            yhat (torch.Tensor): 
                Input tensor of size (B, C, H, W)
            target (torch.Tensor): 
                Target tensor of size (B, H, W), where values 
                of a vector correspond to class index
            target_weight (torch.Tensor): 
                The weight map that points to the pixels
                in clumped nuclei that are overlapping.

        Returns:
        ----------
            torch.Tensor: Computed mse loss (scalar or pixelwise matrix)
        """
        
        target_one_hot = target
        if target.size() != yhat.size():
            target_one_hot = one_hot(target, yhat.shape[1])

        mse =  F.mse_loss(yhat, target_one_hot, reduction="none") # (B, C, H, W)
        mse = torch.mean(mse, dim=1) # to (B, H, W)
        
        if self.class_weights is not None:
            mse = self.apply_class_weights(mse, target)

        if self.edge_weight is not None:
            mse = self.apply_edge_weights(mse, target_weight)
            
        return torch.mean(mse)
