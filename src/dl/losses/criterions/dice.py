import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.dl.utils import one_hot
from .._base._weighted_base_loss import WeightedBaseLoss


class DiceLoss(WeightedBaseLoss):
    def __init__(
            self,
            edge_weight: Optional[float]=None,
            class_weights: Optional[torch.Tensor]=None,
            **kwargs) -> None:
        """
        Sørensen-Dice Coefficient Loss criterion. Optionally applies
        weights at the nuclei edges and weights for different classes.
        
        Args:
        -----------
            edge_weight (float, optional, default=None): 
                Weight to be added to nuclei borders like in Unet paper
            class_weights (torch.Tensor, optional, default=None): 
                Optional tensor of size (n_classes,) for class weights
        """
        super().__init__(class_weights, edge_weight)
        self.eps = 1e-6

    def forward(
            self,
            yhat: torch.Tensor,
            target: torch.Tensor,
            target_weight: Optional[torch.Tensor]=None,
            **kwargs
        ) -> torch.Tensor:
        """
        Computes the DICE coefficient

        Args:
        ---------
            yhat (torch.Tensor): 
                Input tensor of size (B, C, H, W)
            target (torch.Tensro):
                Target tensor of size (B, H, W), where
                values of a vector correspond to class index
            target_weight (torch.Tensor): 
                The weight map that points to the pixels
                in clumped nuclei that are overlapping.

        Returns:
        ---------
            torch.Tensor: computed DICE loss (scalar)
        """
        
        # activation
        yhat_soft = F.softmax(yhat, dim=1)
        target_one_hot = one_hot(target, n_classes=yhat.shape[1])
        assert target_one_hot.shape == yhat.shape

        intersection = torch.sum(yhat_soft * target_one_hot, 1)
        union = torch.sum(yhat_soft + target_one_hot, 1)
        dice = 2.0 * intersection / union.clamp_min(self.eps)
        
        if self.class_weights is not None:
            dice = self.apply_class_weights(dice, target)

        if self.edge_weight is not None:
            dice = self.apply_edge_weights(dice, target_weight)

        return torch.mean(1.0 - dice)