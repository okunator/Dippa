import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.dl.torch_utils import one_hot


# Ported from https://github.com/kornia/kornia/blob/master/kornia/filters/kernels.py
def gaussian(window_size: int, sigma: float, device: torch.device = None) -> torch.Tensor:
    """
    Create a gaussian 1D tensor

    Args:
        window_size (int): number of elements in the tensor
        sigma (float): std of the gaussian distribution
        device (torch.device): device for the tensor

    Returns:
        1D torch.Tensor of length window_size
    """
    x = torch.arange(window_size, device=device).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()


# Ported from https://github.com/kornia/kornia/blob/master/kornia/filters/kernels.py
def gaussian_kernel2d(window_size: int, 
                      sigma: float,
                      n_channels: int = 1,
                      device: torch.device = None) -> torch.Tensor:
    """
    Create 2D window_size**2 sized kernel a gaussial kernel

    Args:
        window_size (int): size of the window
        sigma (float): std of the gaussian distribution
        n_channel (int): number of channels in the image that will be convolved with this kernel
        device (torch.device): device for the kernel

    Returns:
        torch.Tensor of shape (1, 1, win_size, win_size)
    """
    kernel_x = gaussian(window_size, sigma, device=device)
    kernel_y = gaussian(window_size, sigma, device=device)
    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    kernel_2d = kernel_2d.expand(n_channels, 1, window_size, window_size)
    return kernel_2d


# Adapted from https://github.com/ZJUGiveLab/UNet-Version/blob/master/loss/msssimLoss.py
def ssim(yhat: torch.Tensor,
         target: torch.Tensor,
         window_size: int = 11,
         val_range: Optional[int] = None) -> torch.Tensor:
    """
    Computes the structural similarity loss between target and yhat as described in
    UNET3+ paper: https://arxiv.org/pdf/2004.08790.pdf

    Args:
        yhat (torch.Tensor): output from the instance segmentation branch
        target (torch.Tensor): ground truth image
        window_size (int): size of the gaussian kernel

    Returns:
        torch.Tensor computed ssim loss and 
    """
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(target) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(target) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    (_, channel, height, width) = yhat.size()
    target = one_hot(target, n_classes=channel)
    real_size = min(window_size, height, width)
    kernel = gaussian_kernel2d(real_size, sigma=1.5, n_channels=channel, device=yhat.device)

    mu1 = F.conv2d(target, kernel, groups=channel)
    mu2 = F.conv2d(yhat, kernel, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(target * target, kernel, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(yhat * yhat, kernel, groups=channel) - mu2_sq
    sigma12 = F.conv2d(target * yhat, kernel, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map.mean()

    return ret, cs


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
