import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch.nn.modules.loss import _Loss, _WeightedLoss
from src.dl.torch_utils import one_hot


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


# This is adapted from: https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/criterion/ce.html#SymmetricCrossEntropyLoss
# Had to modify the one hot encoding to make it work. Was using F.one_hot that got the dims wrong
# Also added weight maps option.
class WeightedSCELoss(nn.Module):
    """
    The Symmetric Cross Entropy loss.

    It has been proposed in `Symmetric Cross Entropy for Robust Learning
    with Noisy Labels`_.

    .. _Symmetric Cross Entropy for Robust Learning with Noisy Labels:
        https://arxiv.org/abs/1908.06112
    """

    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 1.0,
                 edge_weights: bool = True,
                 class_weights: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        """
        Args:
            alpha(float): corresponds to overfitting issue of CE
            beta(float): corresponds to flexible exploration on the robustness of RCE
            edge_weights (bool): Add weight to nuclei borders like in Unet paper
            class_weights (torch.Tensor): Optional tensor of size n_classes for class weights
        """
        super(WeightedSCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.edge_weights = edge_weights
        self.class_weights = class_weights

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_weight: Optional[torch.Tensor] = None,
                edge_weight: Optional[float] = 1.1,
                eps: Optional[float] = 1e-7,
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
        yhat_soft = F.softmax(yhat, dim=1) + eps
        assert target_one_hot.shape == yhat.shape

        yhat = torch.clamp(yhat_soft, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        if self.class_weights is not None:
            w = self.class_weights.expand([H, W, num_classes]).permute(2, 0, 1)
            out_forward = w*(target_one_hot * torch.log(yhat))
            out_reverse = w*(yhat_soft * torch.log(target_one_hot))
        else:
            out_forward = target_one_hot * torch.log(yhat_soft)
            out_reverse = yhat_soft * torch.log(target_one_hot)
        
        cross_entropy = (-torch.sum(out_forward, dim=1))
        reverse_cross_entropy = (-torch.sum(out_reverse, dim=1))

        if self.edge_weights:
            cross_entropy = (cross_entropy*(edge_weight**target_weight)).mean()
            reverse_cross_entropy = (reverse_cross_entropy*(edge_weight**target_weight)).mean()
        else:
            cross_entropy = cross_entropy.mean()
            reverse_cross_entropy = reverse_cross_entropy.mean()

        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        
        return loss


class DiceLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Sørensen-Dice Coefficient Loss criterion. Optionally applies weights
        at the nuclei edges and weights for different classes.
        """
        super(DiceLoss, self).__init__()

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor,
                eps: float = 1e-7,
                **kwargs) -> torch.Tensor:
        """
        Computes the DICE coefficient

        Args:
            yhat: input tensor of size (B, C, H, W)
            target: target tensor of size (B, H, W), where
                    values of a vector correspond to class index

        Returns:
            torch.Tensor: computed DICE loss (scalar)
        """

        # activation
        yhat_soft = F.softmax(yhat, dim=1)

        # one hot target
        target_one_hot = one_hot(target, n_classes=yhat.shape[1])
        assert target_one_hot.shape == yhat.shape

        # dice components
        intersection = torch.sum(yhat_soft * target_one_hot, (1, 2, 3))
        union = torch.sum(yhat_soft + target_one_hot, (1, 2, 3))

        # dice score
        dice = 2.0 * intersection / union.clamp_min(eps)
        return torch.mean(1.0 - dice)


class IoULoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Intersection over union Loss criterion. Optionally applies weights
        at the nuclei edges and weights for different classes.
        """
        super(IoULoss, self).__init__()

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor,
                eps: float = 1e-7,
                **kwargs) -> torch.Tensor:
        """
        Computes the DICE coefficient

        Args:
            yhat: input tensor of size (B, C, H, W)
            target: target tensor of size (B, H, W), where
                    values of a vector correspond to class index

        Returns:
            torch.Tensor: computed DICE loss (scalar)
        """

        # activation
        yhat_soft = F.softmax(yhat, dim=1)

        # one hot target
        target_one_hot = one_hot(target, n_classes=yhat.shape[1])
        assert target_one_hot.shape == yhat.shape
        
        # iou components
        intersection = torch.sum(yhat_soft * target_one_hot, (1, 2, 3))
        union = torch.sum(yhat_soft + target_one_hot, (1, 2, 3))

        # iou score
        iou = intersection / union.clamp_min(eps)
        return torch.mean(1.0 - iou)


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
