import torch
import torch.nn as nn
from typing import Optional


class WeightedBaseLoss(nn.Module):
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 edge_weight: Optional[float] = None) -> None:
        """
        Base class for a cross entropy based loss where the nuclei edges
        and classes can be given weights

        Args:
        ----------
            Class weights (torch.Tensor, optional, default=None):
                Tensor of shape (C, )
            Edge weight (float, optional, default=None): 
                base weight for nuclei border pixels
        """
        super(WeightedBaseLoss, self).__init__()
        self.class_weights = class_weights
        self.edge_weight = edge_weight
    
    def apply_class_weights(self, 
                            loss_matrix: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
        """
        Multiply pixelwise loss matrix by the class weights
        Note: No normalization

        Args:
        ----------
            Loss_matrix (torch.Tensor): 
                Pixelwise losses tensor of shape (B, H, W)
            Target (torch.Tensor): 
                target mask. Shape (B, H, W)
        """
        # add weights
        weight_mat = self.class_weights[target].to(target.device)
        loss = loss_matrix*weight_mat
        
        return loss

    def apply_edge_weights(self, 
                           loss_matrix: torch.Tensor,
                           weight_map: torch.Tensor) -> torch.Tensor:
        """
        Create edge weights by computing edge_weight**weight_map and add those weights to the
        loss _matrix

        Args:
        ----------
            loss_matrix (torch.Tensor): 
                Pixelwise losses tensor of shape (B, C, H, W)
            weight_map (torch.Tensor):
                Map that points to the pixels that will be weighted. Shape (B, H, W)
        """
        return loss_matrix*self.edge_weight**weight_map