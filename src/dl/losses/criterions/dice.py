import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.utils import one_hot


class DiceLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Sørensen-Dice Coefficient Loss criterion. Optionally applies weights
        at the nuclei edges and weights for different classes.
        """
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, 
                yhat: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Computes the DICE coefficient

        Args:
        ---------
            yhat (torch.Tensor): 
                Input tensor of size (B, C, H, W)
            target (torch.Tensro):
                Target tensor of size (B, H, W), where
                values of a vector correspond to class index

        Returns:
        ---------
            torch.Tensor. computed DICE loss (scalar)
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
        dice = 2.0 * intersection / union.clamp_min(self.eps)
        return torch.mean(1.0 - dice)