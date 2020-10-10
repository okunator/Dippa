import torch
from typing import List, Optional
from torch import nn
from torch.nn.modules.loss import _Loss
from src.dl.torch_utils import one_hot

# adapted these from:
# https: // github.com/BloodAxe/pytorch-toolbelt/blob/master/pytorch_toolbelt/losses/joint_loss.py
class WeightedCELoss(_Loss):
    def __init__(self, 
                 weight: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None):
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
                edge_weight: float,
                device: Optional[torch.device] = None) -> torch.Tensor:

        return self.bCE(yhat_inst, target_inst, edge_weight, target_weight) + \
            self.mCE(yhat_type, target_type, edge_weight, target_weight)


# This is ported from: https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/criterion/ce.html#SymmetricCrossEntropyLoss
# Had to modify the one hot encoding to make it work. Was using F.one_hot that got the dims wrong
# Also added reduction option so weight maps can be applied.
class SymmetricCrossEntropyLoss(nn.Module):
    """
    The Symmetric Cross Entropy loss.

    It has been proposed in `Symmetric Cross Entropy for Robust Learning
    with Noisy Labels`_.

    .. _Symmetric Cross Entropy for Robust Learning with Noisy Labels:
        https://arxiv.org/abs/1908.06112
    """


    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 reduction: bool = True):
        """
        Args:
            alpha(float): corresponds to overfitting issue of CE
            beta(float): corresponds to flexible exploration on the robustness of RCE
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction


    def forward(self, 
                input_: torch.Tensor,
                target: torch.Tensor,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """Calculates loss between ``input_`` and ``target`` tensors.

        Args:
            input_: input tensor of size (B, C, H, W)
            target: target tensor of size (batch_size), where
                    values of a vector correspond to class index
            device: the used one_hot function needs to know the device as input

        Returns:
            torch.Tensor: computed loss
        """
        num_classes = input_.shape[1]
        target_one_hot = one_hot(target, num_classes, device)
        assert target_one_hot.shape == input_.shape

        input_ = torch.clamp(input_, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        if self.reduction: 
            cross_entropy = (-torch.sum(target_one_hot * torch.log(input_), dim=1)).mean()
            reverse_cross_entropy = (-torch.sum(input_ * torch.log(target_one_hot), dim=1)).mean()
        else:
            cross_entropy = (-torch.sum(target_one_hot*torch.log(input_), dim=1))
            reverse_cross_entropy = (-torch.sum(input_*torch.log(target_one_hot), dim=1))

        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        return loss


class WeightedSymmetricCELoss(_Loss):
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 reduction: bool = True,
                 weight: float = 1.0):
        """
        A weighted SCE loss with an option to add weight maps to the borders of nucleis
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.loss = SymmetricCrossEntropyLoss(alpha, beta, reduction)

    def forward(self,           
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_weight: torch.Tensor,
                edge_weight: int,
                device: Optional[torch.device] = None) -> torch.Tensor:

        if self.reduction:
            loss = self.loss(yhat, target, device)
        else:
            loss_matrix = self.loss(yhat, target, device)
            loss = (loss_matrix * (edge_weight**target_weight)).mean()

        return loss * self.weight


class JointSymmetricCELoss(_Loss):
    def __init__(self,
                 first_weight: float = 1.0,
                 second_weight: float = 1.0,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 reduction: bool = True,
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
        self.mCE = WeightedSymmetricCELoss(alpha, beta, reduction, second_weight)

    def forward(self,
                yhat_inst: torch.Tensor,
                yhat_type: torch.Tensor,
                target_inst: torch.Tensor,
                target_type: torch.Tensor,
                target_weight: torch.Tensor,
                edge_weight: float,
                device: Optional[torch.device]= None) -> torch.Tensor:

        return self.bCE(yhat_inst, target_inst, edge_weight, target_weight) + \
            self.mCE(yhat_type, target_type, edge_weight, target_weight, device)





