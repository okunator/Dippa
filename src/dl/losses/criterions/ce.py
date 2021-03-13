import torch
import torch.nn as nn
from typing import List, Optional
from src.dl.losses.weighted_base_loss import WeightedBaseLoss


class WeightedCELoss(WeightedBaseLoss):
    def __init__(self, 
                 edge_weight: Optional[float] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        """
        Wrapper class around CE loss function that applies weights with fixed factor.
        This class adds nuclei border weights to the final computed loss

        Args:
        ---------
            edge_weight (float, optional, default=none): 
                Weight to be added to nuclei borders like in Unet paper
            class_weights (torch.Tensor, optional, default=None): 
                Optional tensor of size (n_classes,) for class weights
        """
        super().__init__(class_weights, edge_weight)
        self.loss = nn.CrossEntropyLoss(
            reduction="none",
            weight=class_weights
        )

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor, 
                target_weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Computes the cross entropy loss

        Args:
        ----------
            yhat (torch.Tensor): 
                The feature map generated from the forward() of the model
            target (torch.Tensor): 
                the ground truth annotations of the input patch
            target_weight (torch.Tensor, default=None): 
                The weight map that points the pixels in clumped nuclei
                That are overlapping.

        Returns:
        ----------
            torch.Tensor:
                computed CE loss (scalar)
        """
        loss = self.loss(yhat, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)
 
        return loss.mean()
