import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional


def ndarray_to_tensor(
        array: np.ndarray, 
        dim_order: str="HWC", 
        add_channel: bool=False
    ) -> torch.Tensor:
    """
    Convert img or mask of shape (H, W)|(H, W, C)|(B, H, W, C) to tensor
    of shape (B, C, H, W)

    Args:
    -----------
        array (np.ndarray): 
            numpy matrix of shape (H, W) or (H, W, C)
        dim_order (str, default="HWC"):
            The order of the dimensions in the tensor

    Returns:
    -----------
        torch.Tensor: A tensor of shape (B, C, H, W).
    """
    assert isinstance(array, np.ndarray), (
        f"Input type: {type(array)} is not np.ndarray"
    )
    assert 1 < len(array.shape) <= 4, (
        f"ndarray.shape {array.shape}, rank needs to be bw [2, 4]"
    )
    assert dim_order in ("HW", "HWC", "BHWC", "BCHW", "BHW")

    # Add channel dim if needed
    if add_channel:
        if dim_order == "HW":
            array = array[None, ...]
        elif dim_order ==  "BHW":
            array = array[:, None, ...]

    # Add batch dim if needed
    if dim_order not in ("BHWC", "BCHW", "BHW"):
        array = array[None, ...]
    
    if dim_order in ("BHWC", "HWC"):
        array = array.transpose(0, 3, 1, 2)

    return torch.from_numpy(array)


def tensor_to_ndarray(
        tensor: torch.Tensor, 
        channel: Optional[int]=None,
        squeeze: Optional[bool]=False
    ) -> np.ndarray:
    """
    Convert img or network output tensor (B, C, H, W) or (B, H, W) 
    to ndarray of shape (B, H, W, C)|(B, H, W)|(H, W, C)|(H, W)

    Args:
    ------------
        tensor (torch.Tensor): 
            tensor of shapes (B, C, H, W)
        channel (int, optional, default=None): 
            index of the channel dimension. If applied returns
            an array of shape (H, W)|(B, H, W)
        squeeze (bool, optional, default=False): 
            if batch size == 1. Squeeze it out. 
    
    Returns:
    -----------
        np.ndarray: ndarray of shape 
        (B, H, W, C)|(B, H, W)|(H, W, C)|(H, W)
    """
    assert isinstance(tensor, torch.Tensor), (
        f"Input type: {type(tensor)} is not torch.Tensor"
    )
    assert 3 <= tensor.dim() <= 4, (
        "tensor needs to have shape (B, H, W) or (B, C, H, W)", 
        f"Shape of the given tensor: {tensor.shape}"
    )
    
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


def argmax_and_flatten(
        yhat: torch.Tensor, 
        activation: Optional[str]=None
    ) -> torch.Tensor:
    """
    Get an output from a prediction by argmaxing a yhat of shape 
    (B, C, H, W) and flatten the result to tensor of shape (1, n_pixels)
    where each value represents a class for a pixel.

    Args:
    -----------
        yhat (torch.Tensor): 
            Logits or softmaxed tensor of shape (B, C, H, W)
        activation (str, optional, default=None): 
            Apply sigmoid or softmax activation before taking argmax

    Returns:
    -----------
         torch.Tensor: A tensor that can be used as input to different 
         classification metrics. Shape: (H, W)
    """
    if activation is not None:
        assert activation in ("sigmoid", "softmax"), (
            f"activation: {activation} sigmoid and softmax allowed."
        )

        if activation == "sigmoid":
            yhat = torch.sigmoid(yhat)
        elif activation == "softmax":
            yhat = F.softmax(yhat, dim=1)

    return torch.argmax(yhat, dim=1).view(1, -1)


def to_device(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Push torch.Tensor or np.ndarray to GPU if it is available.

    Args:
    -----------
        tensor (torch.Tensor or np.ndarray): 
            multi dim array to be pushed to gpu

    Returns:
    -----------
        torch.Tensor. Same shape as input.
    """
    # TODO: implement other types too
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if torch.cuda.is_available():
        tensor = tensor.type("torch.cuda.FloatTensor")
    else:
        tensor = tensor.type("torch.FloatTensor")
    return tensor


def one_hot(type_map: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Take in a type map of shape (B, H, W) with class indices as values 
    and reshape it into a tensor of shape (B, C, H, W)

    Args:
    -----------
        type_map (torch.Tensor): 
            type map
        n_classes (int): 
            number of classes in type_map

    Returns:
    -----------
        torch.Tensor: A one hot tensor from the type map of shape 
        (B, C, H, W)
    """
    assert type_map.dtype == torch.int64, (
        f"Wrong type_map dtype: {type_map.dtype}. Shld be torch.int64"
    )

    one_hot = torch.zeros(
        type_map.shape[0], n_classes, *type_map.shape[1:], 
        device=type_map.device, dtype=type_map.dtype
    )

    return one_hot.scatter_(
        dim=1, index=type_map.unsqueeze(1), value=1.0
    ) + 1e-7


def filter2D(
        input_tensor: torch.Tensor, 
        kernel: torch.Tensor
    ) -> torch.Tensor:
    """
    Convolves a given kernel on input tensor without losing dimensional 
    shape

    Args:
    ----------
        input_tensor (torch.Tensor): 
            Input image/tensor
        kernel (torch.Tensor):
            Convolution kernel/window

    Returns:
    ----------
        torch.Tensor: The convolved tensor of same shape as the input
    """

    (_, channel, _, _) = input_tensor.size()

    # "SAME" padding to avoid losing height and width
    pad = [
        kernel.size(2) // 2,
        kernel.size(2) // 2,
        kernel.size(3) // 2,
        kernel.size(3) // 2
    ]
    pad_tensor = F.pad(input_tensor, pad, "replicate")

    out = F.conv2d(pad_tensor, kernel, groups=channel)
    return out


def binarize(mask: torch.Tensor) -> torch.Tensor:
    mask[mask > 0] = 1
    
    return mask.type("torch.LongTensor")


# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params


# def count_conv_flop(layer, x):
#     out_h = int(x.size()[2] / layer.stride[0])
#     out_w = int(x.size()[3] / layer.stride[1])
#     delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
#                 out_h * out_w / layer.groups
#     return delta_ops
