import torch
import torch.nn as nn
from typing import List, Optional


class WeightedCELoss(nn.Module):
    def __init__(self, 
                 edge_weights: bool = True,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        """
        Wrapper class around CE loss function that applies weights with fixed factor.
        This class adds nuclei border weights to the final computed loss on a feature map generated
        from a H&E image.

        Args:
            edge_weights (bool): Add weight to nuclei borders like in Unet paper
            class_weights (torch.Tensor): Optional tensor of size n_classes for class weights
        """
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            reduction="none",
            weight=class_weights
        )
        self.edge_weights = edge_weights

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor, 
                target_weight: Optional[torch.Tensor] = None,
                edge_weight: Optional[float] = 1.1,
                **kwargs) -> torch.Tensor:
        """
        Computes the cross entropy loss

        Args:
            yhat (torch.Tensor): The feature map generated from the forward() of the model
            target (torch.Tensor): the ground truth annotations of the input patch
            target_weight (torch.Tensor): The weight map that points the pixels in clumped nuclei
                                          that are overlapping.
            edge_weight (float): weights applied to the nuclei edges: edge_weight^target_weight

        Returns:
            torch.Tensor: computed CE loss (scalar)
        """
        if self.edge_weights:
            loss_matrix = self.loss(yhat, target)
            loss = (loss_matrix * (edge_weight**target_weight)).mean()
        else:
            loss = self.loss(yhat, target).mean()
        return loss
