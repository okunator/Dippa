import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.utils import one_hot


class TverskyLoss(nn.Module):
    def __init__(
            self,
            alpha: float=0.7,
            beta: float=0.3,
            **kwargs
        ) -> None:
        """
        Tversky loss: https://arxiv.org/abs/1706.05721

        Args:
        ---------
            alpha (float, default=0.7):
                 False positive dice coefficient
            beta (float, default=0.3)
                False negative tanimoto coefficient
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
            self,
            yhat: torch.Tensor,
            target: torch.Tensor,
            eps: float=1e-7,
            **kwargs
        ) -> torch.Tensor:
        """
        Computes the tversky loss

        Args:
        ----------
            yhat (torch.Tensor): 
                Input tensor of size (B, C, H, W)
            target (torch.Tensor): 
                Target tensor of size (B, H, W), where values 
                of a vector correspond to class index
                    
        Returns:
        ----------
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