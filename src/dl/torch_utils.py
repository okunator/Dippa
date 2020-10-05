import torch
import numpy as np


def ndarray_to_tensor(ndarray: np.ndarray) -> torch.Tensor:
    """
    Convert img or mask (H, W, C) to tensor (1, C, H, W)
    """
    # Add batch dim
    ndarray = ndarray[None, ...]
    return torch.from_numpy(ndarray.transpose(0, 3, 1, 2))


def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert cuda img or network output tensor (B, C, H, W) to ndarray (H, W, C)
    """
    return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze()


def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
    """
    Push torch.Tensor to GPU if it is available.
    """
    # TODO: implement other types too
    if torch.cuda.is_available():
        tensor = tensor.type("torch.cuda.FloatTensor")
    else:
        tensor = tensor.type("torch.FloatTensor")
    return tensor
