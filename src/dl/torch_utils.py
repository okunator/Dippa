import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional


def ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert img or mask of shape (H, W)|(H, W, C)|(B, H, W, C) to tensor (B, C, H, W)
    Args:
        array (np.ndarray) : numpy matrix of shape (H, W) or (H, W, C)
    """
    assert isinstance(array, np.ndarray), f"Input type: {type(array)} is not np.ndarray"
    assert 1 < len(array.shape) <= 4, f"ndarray.shape {array.shape}, rank needs to be bw [2, 4]" 

    # Add channel dim if needed
    if len(array.shape) == 2:
        array = array[..., None]

    # Add batch dim if needed
    if len(array.shape) < 4:
        array = array[None, ...]

    return torch.from_numpy(array.transpose(0, 3, 1, 2))


def tensor_to_ndarray(tensor: torch.Tensor, 
                      channel: Optional[int] = None,
                      squeeze: Optional[bool] = False) -> np.ndarray:
    """
    Convert img or network output tensor (B, C, H, W) or (B, H, W) to ndarray 
    of shape (B, H, W, C)|(B, H, W)|(H, W, C)|(H, W)

    Args:
        tensor (torch.Tensor): tensor of size (B, C, H, W)
        channel (Optional[int]): index of the channel dimension. If applied returns
                                 an array of shape (H, W)|(B, H, W)
        squeeze (Optional[bool]): if batch size == 1. Squeeze it out. 
    """
    assert isinstance(tensor, torch.Tensor), f"Input type: {type(tensor)} is not torch.Tensor"
    assert 3 <= tensor.dim() <= 4, f"tensor needs to have shape (B, H, W) or (B, C, H, W). Shape {tensor.shape}"
    assert tensor.shape[1] >= 2, f"tensor needs to have at least two channels. shape: {tensor.shape}" 
    

    res_tensor = tensor.detach()
    if tensor.is_cuda:
        res_tensor = res_tensor.cpu()

    if res_tensor.dim() == 4:
        res_tensor = res_tensor.numpy().transpose(0, 2, 3, 1) # (B, H, W, C)
    else:
        res_tensor = res_tensor.numpy().transpose(1, 2, 0) # (H, W, B) 

    if squeeze:
        res_tensor = res_tensor.squeeze() # if B == 1 -> (H, W, C) or (H, W) 

    if channel is not None and len(res_tensor.shape) != 2:
        res_tensor = res_tensor[..., channel] # -> (H, W) or (B, H, W)

    return res_tensor


def argmax_and_flatten(yhat: torch.Tensor, activation: Optional[str] = None) -> torch.Tensor:
    """
    Get an output from a prediction by argmaxing a yhat of shape (B, C, H, W)
    and flatten the result to tensor of shape (1, n_pixels). Where each value represents
    a class for a pixel.
    Args:
        yhat (torch.Tensor): logits or softmaxed tensor of shape (B, C, H, W)
        activation (Optional[str]): apply sigmoid or softmax activation before taking argmax
    Returns:
         a tensor that can be inputted to different classification metrics. Shape (H, W)
    """
    if activation is not None:
        assert activation in ("sigmoid", "softmax"), f"activation: {activation} sigmoid and softmax allowed."
        if activation == "sigmoid":
            yhat = torch.sigmoid(yhat)
        elif activation == "softmax":
            yhat = F.softmax(yhat, dim=1)

    return torch.argmax(yhat, dim=1).view(1, -1)


def to_device(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Push torch.Tensor or np.ndarray to GPU if it is available.
    """
    # TODO: implement other types too
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if torch.cuda.is_available():
        tensor = tensor.type("torch.cuda.FloatTensor")
    else:
        tensor = tensor.type("torch.FloatTensor")
    return tensor


# Adapted from: https: // kornia.readthedocs.io/en/latest/_modules/kornia/utils/metrics/confusion_matrix.html
def confusion_mat(yhat: torch.Tensor, 
                  target: torch.Tensor, 
                  activation: Optional[str] = None) -> torch.Tensor:
    """
    Computes confusion matrix from the soft mask and target tensor

    Args:
        yhat (torch.Tensor): the soft mask from the network of shape (B, C, H, W)
        target (torch.Tensor): the target matrix of shape (B, H, W)
        activation (Optional[str]): apply sigmoid or softmax activation before taking argmax

    Returns:
        torch.Tensor of shape (B, num_classes, num_classes)
    """

    if activation is not None:
        assert activation in (
            "sigmoid", "softmax"), f"activation: {activation} sigmoid and softmax allowed."
        if activation == "sigmoid":
            yhat_soft = torch.sigmoid(yhat)
        elif activation == "softmax":
            yhat_soft = F.softmax(yhat, dim=1)
        else:
            yhat_soft = yhat
    
    n_classes = yhat_soft.shape[1]
    batch_size = yhat_soft.shape[0]
    bins = target + torch.argmax(yhat_soft, dim=1)*n_classes
    bins_vec = bins.view(batch_size, -1)

    confusion_list = []
    for iter_id in range(batch_size):
        pb = bins_vec[iter_id]
        bin_count = torch.bincount(pb, minlength=n_classes**2)
        confusion_list.append(bin_count)

    confusion_vec = torch.stack(confusion_list)
    confusion_mat = confusion_vec.view(batch_size, n_classes, n_classes).to(torch.float32)

    return confusion_mat


# from https://kornia.readthedocs.io/en/latest/_modules/kornia/utils/one_hot.html#one_hot
def one_hot(type_map: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Take in a type map of shape (B, H, W) with class indices as values and reshape it
    into a tensor of shape (B, C, H, W)

    Args:
        type_map (torch.Tensor): type map
        n_classes (int): number of classes in type_map

    Returns:
        torch.Tensor onet hot tensor from the type map of shape (B, C, H, W)
    """
    assert type_map.dtype == torch.int64, f"Wrong type_map dtype: {type_map.dtype}. Should be torch.int64"
    one_hot = torch.zeros(type_map.shape[0], n_classes, *type_map.shape[1:], device=type_map.device, dtype=type_map.dtype)
    return one_hot.scatter_(dim=1, index=type_map.unsqueeze(1), value=1.0) + 1e-6


# from: https://kornia.readthedocs.io/en/latest/_modules/kornia/utils/metrics/mean_iou.html#mean_iou
def mean_iou(yhat: torch.Tensor, 
             target: torch.Tensor,
             activation: Optional[str] = None,
             eps: Optional[float] = 1e-7) -> torch.Tensor:
    """
    Compute the mean iou for each class in the segmented image

    Args:
        yhat (torch.Tensor): the soft mask from the network of shape (B, C, H, W)
        target (torch.Tensor): the target matrix of shape (B, H, W)
        activation (Optional[str]): apply sigmoid or softmax activation before taking argmax

    Returns:
        torch.Tensor of shape (B, num_classes, num_classes)
    """
    conf_mat = confusion_mat(yhat, target, activation)
    sum_over_row = torch.sum(conf_mat, dim=1)
    sum_over_col = torch.sum(conf_mat, dim=2)
    conf_mat_diag = torch.diagonal(conf_mat, dim1=-2, dim2=-1)
    denominator = sum_over_row + sum_over_col - conf_mat_diag

    # NOTE: we add epsilon so that samples that are neither in the
    # prediction or ground truth are taken into account.
    ious = (conf_mat_diag + eps) / (denominator + eps)
    return ious


def sobel_hv(tensor: torch.Tensor, kernel_size: int = 5, direction: str = "x"):
    """
    Computes first order derviatives in x or y direction using same sobel kernel
    as in the HoVer-net paper.   

    Args:
        tensor (torch.Tensor): input tensor. Shape (B, 1, H, W) or (B, H, W)
        kernel_size (int): size of the convolution kernel
        direction (str): direction of the derivative. One of ("x", "y")

    Returns:
        torch.Tensor computed 1st order derivatives of the input tensor. Shape (B, 2, H, W)
    """

    # Add channel dimension if shape (B, H, W)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    assert tensor.shape[1] == 1, f"Input tensor shape expected to have shape (B, H, W) or (B, 1, H, W). Got: {tensor.shape}" 
    assert kernel_size % 2 == 1, f"size must be odd. size: {kernel_size}"
    assert direction in ("x", "y"), "direction need to be one of ('x', 'y')"

    # Generate the sobel kernels
    range_h = torch.arange(-kernel_size//2+1, kernel_size//2+1, dtype=torch.float32, device=tensor.device)
    range_v = torch.arange(-kernel_size//2+1, kernel_size//2+1, dtype=torch.float32, device=tensor.device)
    h, v = torch.meshgrid(range_h, range_v)

    if direction == "x":
        kernel = h / (h*h + v*v + 1e-7)
        kernel = kernel.flip(0).unsqueeze(0).unsqueeze(0)
    elif direction == "y":
        kernel = v / (h*h + v*v + 1e-7)
        kernel = kernel.flip(1).unsqueeze(0).unsqueeze(0)

    # "SAME" padding to avoid losing height and width
    pad = [
        kernel.size(2) // 2,
        kernel.size(2) // 2,
        kernel.size(3) // 2,
        kernel.size(3) // 2
    ]
    pad_tensor = F.pad(tensor, pad, "replicate")

    # Compute the gradient
    grad = F.conv2d(pad_tensor, kernel)
    return grad


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