import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.utils import one_hot


class IoULoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        Intersection over union Loss criterion. Optionally applies 
        weights at the nuclei edges and weights for different classes.
        """
        super(IoULoss, self).__init__()
        self.eps = 1e-6

    def forward(
            self,
            yhat: torch.Tensor,
            target: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:
        """
        Computes the DICE coefficient

        Args:
        ---------
            yhat (torch.Tensor): 
                Input tensor of size (B, C, H, W)
            target (torch.Tensor): 
                Target tensor of size (B, H, W), where values 
                of a vector correspond to class index

        Returns:
        ---------
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
        iou = intersection / union.clamp_min(self.eps)
        return torch.mean(1.0 - iou)
