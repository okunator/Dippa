import numpy as np
import albumentations as A
from typing import List
from albumentations.core.transforms_interface import BasicTransform

# TODO add docs

def tta_augs() -> List[BasicTransform]:
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