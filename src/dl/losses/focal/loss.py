import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from src.dl.torch_utils import one_hot


class WeightedFocalLoss(nn.Module):
    """
    Focal loss criterion: https://arxiv.org/abs/1708.02002

    Args:
        alpha (float): weight factor b/w [0,1]
        gamma (float): focusing factor
    """

    def __init__(self,
                 alpha: float = 0.5,
                 gamma: float = 2.0,
                 edge_weights: bool = True,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> None:

        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.edge_weights = edge_weights
        self.class_weights = class_weights

    def forward(self,
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None,
                edge_weight: Optional[float] = 1.1,
                eps: float = 1e-7,
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

        input_soft = F.softmax(yhat, dim=1) + eps
        H = yhat.shape[2]
        W = yhat.shape[3]
        num_classes = yhat.shape[1]
        target_one_hot = one_hot(target, num_classes)
        assert target_one_hot.shape == yhat.shape

        weight = (1.0 - input_soft)**self.gamma
        focal = self.alpha * weight * torch.log(input_soft)

        if self.class_weights is not None:
            w = self.class_weights.expand([H, W, num_classes]).permute(2, 0, 1)
            loss_temp = -torch.sum(w*(target_one_hot * focal), dim=1)
        else:
            loss_temp = -torch.sum(target_one_hot * focal, dim=1)

        if self.edge_weights:
            loss = (loss_temp*(edge_weight**target_weight)).mean()
        else:
            loss = loss_temp.mean()

        return loss