import numpy as np
import albumentations as A
from typing import List
from albumentations.core.transforms_interface import BasicTransform


def tta_augs() -> List[BasicTransform]:
    """
    Returns a list of flip and rotation augmentations
    """
    return [
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1),
        A.Rotate((45, 45), p=1),
        A.Rotate((95, 95), p=1),
        A.Rotate((155, 155), p=1),
        A.Rotate((245, 245), p=1),
    ]


def tta_deaugs() -> List[BasicTransform]:
    """
    Returns a list of rotations and flips that are used as deaugmentations for tta_augs()
    """
    return [
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1),
        A.Rotate((-45, -45), p=1),
        A.Rotate((-95, -95), p=1),
        A.Rotate((-155, -155), p=1),
        A.Rotate((-245, -245), p=1),
    ]


def tta_five_crops(io: np.ndarray) -> List[BasicTransform]:
    """
    returns a list of five crops of same size on an image. One at each corner and a center crop

    Args:
        io (np.ndarray): input image. shape (H, W)
    """
    return [
        # left upper crop
        A.Crop(0, 0, io.shape[0]//2, io.shape[1]//2),
        #right upper crop
        A.Crop(io.shape[0]//2, io.shape[1]//2, io.shape[0], io.shape[1]),
        # left lower crop
        A.Crop(0, io.shape[0]//2, io.shape[1]//2, io.shape[1]),
        # right lower crop
        A.Crop(io.shape[0]//2, 0, io.shape[1], io.shape[0]//2),
        # Center crop
        A.Crop(
            io.shape[0]//2//2, 
            io.shape[0]//2//2, 
            io.shape[0]//2+io.shape[0]//2//2,
            io.shape[1]//2+io.shape[1]//2//2
        )
    ]


def resize(height: int, width: int, **kwargs) -> List[BasicTransform]:
    """
    Wrapper for albumentations resize transform. 

    Args:
        height (int): height of the output image
        width (int): width of the input image

    Returns:
        A resize transform
    """
    return  A.Resize(height=height, width=width, p=1)