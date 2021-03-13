import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from src.dl.torch_utils import one_hot
from src.dl.losses.weighted_base_loss import WeightedBaseLoss


# This is adapted from: https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/criterion/ce.html#SymmetricCrossEntropyLoss
class WeightedSCELoss(WeightedBaseLoss):
    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 1.0,
                 edge_weight: Optional[float] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        """
        The Symmetric Cross Entropy loss: https://arxiv.org/abs/1908.06112

        Args:
            alpha(float): corresponds to overfitting issue of CE
            beta(float): corresponds to flexible exploration on the robustness of RCE
            edge_weight (float, optional): weight to be added to nuclei borders like in Unet paper
            class_weights (torch.Tensor, optional): Optional tensor of size (n_classes,) for class weights
        """
        super(WeightedSCELoss, self).__init__(class_weights, edge_weight)
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Computes the symmetric cross entropy loss between ``yhat`` and ``target`` tensors.

        Args:
            yhat: input tensor of size (B, C, H, W)
            target: target tensor of size (B, H, W), where
                    values of a vector correspond to class index
            target_weight (torch.Tensor): The weight map that points the pixels in clumped nuclei
                                that are overlapping.
            edge_weight (float): weights applied to the nuclei edges: edge_weight^target_weight

        Returns:
            torch.Tensor: computed SCE loss (scalar)
        """
        H = yhat.shape[2]
        W = yhat.shape[3]
        num_classes = yhat.shape[1]
        target_one_hot = one_hot(target, num_classes)
        yhat_soft = F.softmax(yhat, dim=1) + self.eps
        assert target_one_hot.shape == yhat.shape

        yhat = torch.clamp(yhat_soft, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        forward = target_one_hot * torch.log(yhat_soft)
        reverse = yhat_soft * torch.log(target_one_hot)

        cross_entropy = (-torch.sum(forward, dim=1))
        reverse_cross_entropy = (-torch.sum(reverse, dim=1))

        if self.class_weights is not None:
            cross_entropy = self.apply_class_weights(cross_entropy, target)
            reverse_cross_entropy = self.apply_class_weights(reverse_cross_entropy, target)    

        if self.edge_weight is not None:
            cross_entropy = self.apply_edge_weights(cross_entropy, target_weight)
            reverse_cross_entropy = self.apply_edge_weights(reverse_cross_entropy, target_weight)

        loss = self.alpha * cross_entropy.mean() + self.beta * reverse_cross_entropy.mean()
        
        return loss
