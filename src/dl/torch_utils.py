import torch
import numpy as np
from torch import nn
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
    Convert img or network output tensor (B, C, H, W) to ndarray 
    of shape (B, H, W, C)|(B, H, W)|(H, W, C)|(H, W)

    Args:
        tensor (torch.Tensor): tensor of size (B, C, H, W)
        channel (Optional[int]): index of the channel dimension. If applied returns
                                 an array of shape (H, W)|(B, H, W)
        squeeze (Optional[bool]): if batch size == 1. Squeeze it out. 
    """
    assert isinstance(tensor, torch.Tensor), f"Input type: {type(tensor)} is not torch.Tensor"
    assert tensor.shape[1] >= 2, f"tensor needs to have at least two channels. shape: {tensor.shape}" 

    res_tensor = tensor.detach()
    if tensor.is_cuda:
        res_tensor = res_tensor.cpu()

    res_tensor = res_tensor.numpy().transpose(0, 2, 3, 1) # (B, H, W, C)

    if squeeze:
        res_tensor = res_tensor.squeeze() # if B == 1 -> (H, W, C) 

    if channel is not None:
        res_tensor = res_tensor[..., channel] # -> (H, W) or (B, H, W)

    return res_tensor


def argmax_and_flatten(pred_map: torch.Tensor) -> torch.Tensor:
    """
    Get an output from a prediction by argmaxing a pred_map of shape (B, C, H, W)
    and flatten the result to tensor of shape (1, n_pixels). Where each value represents
    a class for a pixel.
    Args:
        pred_map (torch.Tensor): logits or softmaxed tensor of shape (B, C, H, W)
    Returns:
         a tensor that can be inputted to different classification metrics 
    """
    return torch.argmax(pred_map, dim=1).view(1, -1)


def thresh_and_flatten(pred_map: torch.Tensor) -> torch.Tensor:
    pass


def argmax():
    pass


def one_hot(type_map: torch.Tensor,
            n_classes: int, 
            device: Optional[torch.device]) -> torch.Tensor:
    """
    Take in a type map of shape (B, H, W) with class indices as values and reshape it
    into a tensor of shape (B, C, H, W)
    Args:
        type_map (torch.Tensor): type map
        n_classes (int): number of classes in type_map
        device (torch.Device): sets the device for the torch.zeros
    """
    one_hot = torch.zeros(type_map.shape[0], n_classes, *type_map.shape[1:], device=device)
    return one_hot.scatter_(dim=1, index=type_map.unsqueeze(1), value=1.0) + 1e-6


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