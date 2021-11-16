import torch
import numpy as np
from typing import Union


# Dataset level normalization
def normalize(
        img: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        to_uint8: bool=True
    ) -> torch.Tensor:
    """
    Normalize a tensor with mean and standard deviation of the dataset.

    Args:
    ---------
        img (torch.Tensor): 
            Tensor img of Shape (C, H, W) or (B, C, H, W).
        mean (np.ndarray): 
            Means for each channel. Shape (1, 3)
        std (np.ndarray): 
            Standard deviations for each channel. Shape (1, 3)
        to_uint8 (bool, default=True):
            If input tensor values between [0, 255]. The std and mean
            can be scaled from [0, 1] -> [0, 255].

    Returns:
    ----------
        Tensor: Normalized Tensor image. Same shape as input.
    """
    assert isinstance(img, torch.Tensor), (
        f"input img needs to be a  tensor. Got {img.dtype}."
    )
    assert img.ndim >= 3, (
        f"img tensor shae should be either (C, H, W)|(B, C, H, W)"
    )

    if to_uint8:
        mean = mean*255
        std = std*255

    img = img.float()
    mean = torch.as_tensor(mean[0], dtype=img.dtype, device=img.device)
    std = torch.as_tensor(std[0], dtype=img.dtype, device=img.device)

    assert not (std == 0).any(), (
        "zeros detected in std-values -> would lead to zero-div error"
    )

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    
    img.sub_(mean).div_(std)

    return img


# Channel-wise normalizations per image.
# Ported from: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 
              inclusive.

    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted 
    # value indeed corresponds to k=1, not k=0! Use float(q) instead of 
    # q directly, so that ``round()`` returns an integer, even if q is 
    # a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def percentile_normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """ 
    1-99 percentile normalization per image channel. Numpy version

    Args:
    -----------
        img (torch.Tensor):
            Input image to be normalized. Shape (C, H, W)
    
    Returns:
    -----------
        torch.Tensor. Normalized image. Shape (C, H, W).
    """
    C, _, _ = img.shape
    img = img.float()
    percentile1 = torch.zeros(C, dtype=img.dtype, device=img.device)
    percentile99 = torch.zeros(C, dtype=img.dtype, device=img.device)
    for channel in range(C):
        percentile1[channel] = percentile(img[channel, ...], q=1)
        percentile99[channel] = percentile(img[channel, ...], q=99)

    img.sub_(percentile1.view(-1, 1, 1)).div_(
        (percentile99 - percentile1).view(-1, 1, 1)
    )

    return img


def minmax_normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalize image tensor per channel

    Args:
    -----------
        img (torch.Tensor):
            input image tensor. shape (C, H, W).

    Returns:
    -----------
        torch.Tensor. Minmax normalized image tensor. Shape (C, H, W).
    """
    C, _, _ = img.shape
    img = img.float()
    chl_min = torch.zeros(C, dtype=img.dtype, device=img.device)
    chl_max = torch.zeros(C, dtype=img.dtype, device=img.device)
    for channel in range(C):
        chl_min[channel] = torch.min(img[channel, ...])
        chl_max[channel] = torch.max(img[channel, ...])

    img.sub_(chl_min.view(-1, 1, 1)).div_(
        (chl_max - chl_min).view(-1, 1, 1)
    )

    return img


def normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """
    Standardize image tensor per channel

    Args:
    -----------
        img (torch.Tensor):
            input image tensor. shape (C, H, W).

    Returns:
    -----------
        torch.Tensor. Standardized image tensor. Shape (C, H, W).
    """
    img = img.float()
    chl_means = torch.mean(img.float(), dim=(1, 2))
    chl_stds = torch.std(img.float(), dim=(1, 2))

    img.sub_(chl_means.view(-1, 1, 1)).div_(chl_stds.view(-1, 1, 1))
    
    return img
