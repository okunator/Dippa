import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from src.dl.torch_utils import one_hot, sobel_hv, ssim


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


# This is adapted from: https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/criterion/ce.html#SymmetricCrossEntropyLoss
class WeightedSCELoss(nn.Module):
    """
    The Symmetric Cross Entropy loss: https://arxiv.org/abs/1908.06112
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
            forward = w*(target_one_hot * torch.log(yhat))
            reverse = w*(yhat_soft * torch.log(target_one_hot))
        else:
            forward = target_one_hot * torch.log(yhat_soft)
            reverse = yhat_soft * torch.log(target_one_hot)
        
        cross_entropy = (-torch.sum(forward, dim=1))
        reverse_cross_entropy = (-torch.sum(reverse, dim=1))

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
        target_one_hot = one_hot(target, n_classes=yhat.shape[1])
        assert target_one_hot.shape == yhat.shape
        
        # iou components
        intersection = torch.sum(yhat_soft * target_one_hot, (1, 2, 3))
        union = torch.sum(yhat_soft + target_one_hot, (1, 2, 3))

        # iou score
        iou = intersection / union.clamp_min(eps)
        return torch.mean(1.0 - iou)


class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 **kwargs) -> None:
        """
        Tversky loss: https://arxiv.org/abs/1706.05721

        Args:
            fp dice coeff
            fn tanimoto coeff
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                yhat: torch.Tensor,
                target: torch.Tensor,
                eps: float = 1e-7,
                **kwargs) -> torch.Tensor:
        """
        Computes the tversky loss

        Args:
            yhat: input tensor of size (B, C, H, W)
            target: target tensor of size (B, H, W), where
                    values of a vector correspond to class index
                    
        Returns:
            torch.Tensor: computed Tversky loss (scalar)
        """
        yhat_soft = F.softmax(yhat, dim=1)
        target_one_hot = one_hot(target, n_classes=yhat.shape[1])
        assert target_one_hot.shape == yhat.shape

        # compute the actual dice score
        intersection = torch.sum(yhat_soft * target_one_hot, (1, 2, 3))
        fps = torch.sum(yhat_soft*(1.0 - target_one_hot), (1, 2, 3))
        fns = torch.sum(target_one_hot*(1.0 - yhat_soft), (1, 2, 3))

        denom = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = intersection / denom.clamp_min(eps)
        return torch.mean(1.0 - tversky_loss)


class HoVerLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Computes the loss for Horizontal and vertical branch from HoVer-Net.
        See: https://arxiv.org/abs/1812.06499
        """
        super(HoVerLoss, self).__init__()

    def forward(self,
                yhat: torch.Tensor,
                target: torch.Tensor,
                target_inst: torch.Tensor,
                eps: float = 1e-7,
                **kwargs) -> torch.Tensor:
        """
        Computes the HoVer loss. I.e. mse for regressed HoVer maps against GT HoVer maps and
        gradient mse for the same inputs where 1st order sobel derivative is computed on the inputs

        Args:
            yhat (torch.Tensor): input tensor of size (B, 2, H, W). Regressed HoVer map 
            target (torch.Tensor): target tensor of shape (B, 2, H, W). Contains GT HoVer-maps 
            target_inst (torch.Tensor): target for instance segmentation used to focus loss to the
                                        correct nucleis. Shape (B, H, W)

        Returns:
            torch.Tensor: computed HoVer loss (scalar)
        """
        # Compute mse loss
        loss_mse = yhat - target
        loss_mse = (loss_mse*loss_mse).mean()

        # Compute msge loss
        pred_grad_x = sobel_hv(yhat[:, 0, ...], direction="x")
        pred_grad_y = sobel_hv(yhat[:, 1, ...], direction="y")
        pred_grad = torch.stack([pred_grad_x.squeeze(1), pred_grad_y.squeeze(1)], dim=1)

        target_grad_x = sobel_hv(target[:, 0, ...], direction="x")
        target_grad_y = sobel_hv(target[:, 1, ...], direction="y")
        target_grad = torch.stack([target_grad_x.squeeze(1), target_grad_y.squeeze(1)], dim=1)

        focus = torch.stack([target_inst, target_inst], dim=1)
        loss_msge = pred_grad - target_grad
        loss_msge = focus*(loss_msge * loss_msge)
        loss_msge = loss_msge.sum() / focus.clamp_min(eps).sum()

        # Compute the total loss
        loss = loss_msge + loss_mse 
        return loss


# Adapted from https://github.com/ZJUGiveLab/UNet-Version/blob/master/loss/msssimLoss.py
class MSSSIM(nn.Module):
    def __init__(self,
                 window_size: int = 11,
                 **kwargs) -> None:
        """
        MSSIM loss from UNET3+ paper: https://arxiv.org/pdf/2004.08790.pdf
        to penalize fuzzy boundaries

        Args:
            window_size (int): size of the gaussian kernel
        """
        super(MSSSIM, self).__init__()
        self.window_size = window_size

    def forward(self,
                yhat: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Computes the MS-SSIM loss

        Args:
            yhat (torch.Tensor): output from the instance segmentation branch
            target (torch.Tensor): ground truth image
        
        Returns 
            Computed MS-SSIM Loss
        """
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=yhat.device)
        levels = weights.size()[0]
        msssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = ssim(yhat, target, window_size=self.window_size, val_range=None)
            msssim.append(sim)
            mcs.append(cs)

            yhat = F.avg_pool2d(yhat, (2, 2))
            target = F.avg_pool2d(target.float(), (2, 2)).long()

        msssim = torch.stack(msssim)
        mcs = torch.stack(mcs)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        msssim = (msssim + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = msssim ** weights
        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        loss = torch.prod(pow1[:-1] * pow2[-1])
        return loss 

