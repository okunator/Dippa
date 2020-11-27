import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from src.dl.torch_utils import one_hot
from src.dl.losses.weighted_base_loss import WeightedBaseLoss


class WeightedFocalLoss(WeightedBaseLoss):
    """
    Focal loss criterion: https://arxiv.org/abs/1708.02002

    Args:
        alpha (float): weight factor b/w [0,1]
        gamma (float): focusing factor
        edge_weight (float, optional): weight to be added to nuclei borders like in Unet paper
        class_weights (torch.Tensor, optional): Optional tensor of size (n_classes,) for class weights
    """

    def __init__(self,
                 alpha: float = 0.5,
                 gamma: float = 2.0,
                 edge_weight: Optional[float] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> None:

        super(WeightedFocalLoss, self).__init__(class_weights, edge_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6

    def forward(self,
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Computes the focal loss. Option to apply nuclei borders weights and class weights

        Args:
            yhat: input tensor of size (B, C, H, W)
            target: target tensor of size (B, H, W), where
                    values of a vector correspond to class index
            target_weight (torch.Tensor): The weight map that points the pixels in clumped nuclei
                           that are overlapping.
            edge_weight (float): weights applied to the nuclei edges: edge_weight^target_weight

        Returns:
            torch.Tensor: computed focal loss (scalar)
        """

        input_soft = F.softmax(yhat, dim=1) + self.eps
        num_classes = yhat.shape[1]
        target_one_hot = one_hot(target, num_classes)
        assert target_one_hot.shape == yhat.shape

        weight = (1.0 - input_soft)**self.gamma
        focal = self.alpha * weight * torch.log(input_soft)
        focal = target_one_hot * focal

        loss = -torch.sum(focal, dim=1)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()