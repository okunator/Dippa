import torch
from typing import List, Optional
from torch import nn
from torch.nn.modules.loss import _Loss
from catalyst.contrib.nn.criterion.ce import SymmetricCrossEntropyLoss

# adapted these from:
# https: // github.com/BloodAxe/pytorch-toolbelt/blob/master/pytorch_toolbelt/losses/joint_loss.py
class WeightedCELoss(_Loss):
    def __init__(self, weight: float = 1.0, class_weights: Optional[torch.Tensor] = None):
        """
        Wrapper class around CE loss function that applies weights with fixed factor.
        This class adds nuclei border weights to the final computed loss on a feature map generated
        from a H&E image.
        Args:
            weight (float): weight added to the reduced loss
            class_weights (torch.Tensor): Optional tensor of size n_classes for class weights
        """
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            reduction="none",
            weight=class_weights
        )

        self.weight = weight

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor, 
                target_weight: torch.Tensor,
                edge_weight: int) -> torch.Tensor:
        """
        Args:
            yhat (torch.Tensor): The feature map generated from the forward() of the model
            target (torch.Tensor): the ground truth annotations of the input patch
            target_weight (torch.Tensor): The weight map that points the pixels in clumped nuclei
                                          that are overlapping.
            edge_weight (int): weights applied to the nuclei edges: edge_weight^target_weight
        """
        loss_matrix = self.loss(yhat, target)
        loss = (loss_matrix * (edge_weight**target_weight)).mean()
        return loss * self.weight


class JointCELoss(_Loss):
    def __init__(self,
                 first_weight: float = 1.0, 
                 second_weight: float = 1.0,
                 class_weights_binary: Optional[torch.Tensor] = None,
                 class_weights_types: Optional[torch.Tensor] = None):
        """
        Adds two weighted CE losses to one joint loss. When a classification decoder branch is added
        to any network, this loss will take in the instance branch and classification branch outputs
        and computes a weighted joint loss for the whole network to ensure convergence
        Args:
            first_weight (float): weight to apply to instance segmentation CE loss
            second_weight (float): weight to apply to the type segmentation branch
            class_weights_binary (torch.Tensor): weights applied to binary classes
            class_weights_types (torch.Tensor): weights applied to cell types
        """
        super().__init__()
        self.bCE = WeightedCELoss(first_weight, class_weights_binary)
        self.mCE = WeightedCELoss(second_weight, class_weights_types)

    def forward(self,
                yhat_inst: torch.Tensor,
                yhat_type: torch.Tensor,
                target_inst: torch.Tensor,
                target_type: torch.Tensor,
                target_weight: torch.Tensor,
                edge_weight: float) -> torch.Tensor:

        return self.bCE(yhat_inst, target_inst, edge_weight, target_weight) + \
            self.mCE(yhat_type, target_type, edge_weight, target_weight)


class WeightedSymmetricCELoss(_Loss):
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 1.0, 
                 weight: float = 1.0):
        """
        Weighted Symmetric CE loss. Catalyst implementation
        Article: https://arxiv.org/abs/1908.06112
        Source at: https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/nn/criterion/ce.py
        """
        super().__init__()
        self.loss = SymmetricCrossEntropyLoss(alpha, beta)
        self.weight = weight
    
    def forward(self, *input) -> torch.Tensor:
        return self.loss(*input) * self.weight



class JointSymmetricCELoss(_Loss):
    def __init__(self,
                 first_weight: float = 1.0,
                 second_weight: float = 1.0,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 class_weights_binary: Optional[torch.Tensor] = None):
        """
        Adds one weighted symmetric CE loss and weighted CE loss to one joint loss. 
        When a classification decoder branch is added to any network, this loss will take
        in the instance branch and classification branch outputs andd computes a weighted
        joint loss for the whole network to ensure convergence symmetricCE takes care of the
        classification branch.
        """
        super().__init__()
        self.bCE = WeightedCELoss(first_weight, class_weights_binary)
        self.mCE = WeightedSymmetricCELoss(alpha=alpha, beta=beta, weight=second_weight)

    def forward(self,
                yhat_inst: torch.Tensor,
                yhat_type: torch.Tensor,
                target_inst: torch.Tensor,
                target_type: torch.Tensor,
                target_weight: torch.Tensor,
                edge_weight: float) -> torch.Tensor:

        return self.bCE(yhat_inst, target_inst, edge_weight, target_weight) + \
            self.mCE(yhat_type, target_type)



