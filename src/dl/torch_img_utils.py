import torch
from typing import Union


# Channel-wise normalizations per image. Normalizations per full training data not yet implemented.

# Ported from: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
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
        torch.Tensor=Normalized image
    """
    C, _, _ = img.shape
    img = img.float()
    percentile1 = torch.zeros(C, dtype=img.dtype, device=img.device)
    percentile99 = torch.zeros(C, dtype=img.dtype, device=img.device)
    for channel in range(C):
        percentile1[channel] = percentile(img[channel, ...], q=1)
        percentile99[channel] = percentile(img[channel, ...], q=99)

    img.sub_(percentile1.view(-1, 1, 1)).div_((percentile99 - percentile1).view(-1, 1, 1))
    return img


def minmax_normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalize image tensor per channel

    Args:
    -----------
        img (torch.Tensor):
            input image tensor. shape (C, H, W).
    """
    C, _, _ = img.shape
    img = img.float()
    chl_min = torch.zeros(C, dtype=img.dtype, device=img.device)
    chl_max = torch.zeros(C, dtype=img.dtype, device=img.device)
    for channel in range(C):
        chl_min[channel] = torch.min(img[channel, ...])
        chl_max[channel] = torch.max(img[channel, ...])

    img.sub_(chl_min.view(-1, 1, 1)).div_((chl_max - chl_min).view(-1, 1, 1))
    return img


def normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """
    Standardize image tensor per channel

    Args:
    -----------
        img (torch.Tensor):
            input image tensor. shape (C, H, W).
    """
    img = img.float()
    chl_means = torch.mean(img1.float(), dim=(1, 2))
    chl_stds = torch.std(img1.float(), dim=(1, 2))

    img.sub_(chl_means.view(-1, 1, 1)).div_(chl_stds.view(-1, 1, 1))
    return img