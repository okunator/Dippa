import cv2
import numpy as np
import albumentations as A
from typing import List, Tuple
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform


def rigid_transforms(**kwargs) -> List[BasicTransform]:
    """
    Wrapper for rigid albumentations augmentations. For every patch, either:
    - Rotation
    - random rotate 90 degrees
    - flip (rotate 180 degrees)
    - transpose (flip x and y axis)
    is applied with a probability of 0.7*(0.5/(0.5+0.5+0.5+0.5))=0.175

    Returns:
    -----------
        A List of possible data augmentations
    """
    return [
        A.OneOf([
            A.Rotate(p=0.8),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
        ], p=1.0)
    ]


def non_rigid_transforms(**kwargs) -> List[BasicTransform]:
    """
    Wrapper for non rigid albumentations augmentations. For every patch, either:
    - elastic transformation
    - grid distortion
    - optical distortion
    is applied with a probability of 0.3*(0.5/(0.5+0.5+0.5))=0.1

    Returns:
    -----------
        A List of possible data augmentations
    """
    return [
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
        ], p=0.3)
    ]


def hue_saturation_transforms(**kwargs) -> List[BasicTransform]:
    """
    Wrapper for non hue saturation albumentations augmentations. For every patch, either:
    - hue saturation value shift
    is applied with a probability of 0.5

    Returns:
    ----------
        A List of possible data augmentations
    """
    return [A.HueSaturationValue(hue_shift_limit=(0,15), sat_shift_limit=0, val_shift_limit=0, p=0.25)]


def blur_transforms(**kwargs) -> List[BasicTransform]:
    """
    Wrapper for blur albumentations augmentations. For every patch, either:
    - motion blur
    - median blur
    - gaussian blur
    is applied with a probability of 0.7*(0.5/(0.5+0.5+0.5))=0.233

    Returns:
    -----------
        A List of possible data augmentations
    """
    return [
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.7)
    ]


def non_spatial_transforms(**kwargs) -> List[BasicTransform]:
    """
    Wrapper for non spatial albumentations augmentations. For every patch, either:
    - CLAHE
    - brightness contrast
    - random gamma
    is applied with a probability of 0.7*(0.5/(0.5+0.5+0.5))=0.233

    Returns:
    ----------
        A List of possible data augmentations
    """
    return [
        A.OneOf([
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5)
        ], p=0.7)
    ]


def center_crop(height: int, width: int, **kwargs) -> List[BasicTransform]:
    """
    Wrapper for albumentations center crop. For every patch
    a crop a is extracted from the center, p=1.0.

    Args:
    ----------
        height (int): height of the output image
        width (int): width of the input image

    Returns:
    ----------
        A center crop transform
    """
    return [A.CenterCrop(height=height, width=width, always_apply=True, p=1)]


def random_crop(height: int, width: int, **kwargs) -> List[BasicTransform]:
    """
    Wrapper for albumentations random crop. For every patch
    a crop a is extracted randomly, p=1.0.

    Args:
    ----------
        height (int): height of the output image
        width (int): width of the input image

    Returns:
    ----------
        A random crop transform
    """
    return [A.RandomCrop(height=height, width=width, always_apply=True, p=1)]


def resize(height: int, width: int, **kwargs) -> List[BasicTransform]:
    """
    Wrapper for albumentations resize transform. 

    Args:
        height (int): height of the output image
        width (int): width of the input image

    Returns:
    ----------
        A resize transform
    """
    return  [A.Resize(height=height, width=width, p=1)]


def normalize(**kwargs):
    return [A.Normalize()]


def to_tensor(**kwargs) -> List[BasicTransform]:
    """
    Wrapper for albumentations to tensor func. For every patch
    (np.ndarray) is converted into torch.Tensor

    Returns:
    ----------
        A tensor conversion transform
    """
    return [ToTensorV2()]


def compose(transforms_to_compose: List[BasicTransform]) -> A.Compose:
    """
    Wrapper for albumentations compose func. Takes in a list of albumentation
    transforms and composes them to one transformation pipeline

    Returns:
    ----------
        A composed pipeline of albumentation transforms
    """
    result = A.Compose([item for sublist in transforms_to_compose for item in sublist])
    return result


###################################
###################################


def rigid_augs_and_crop(image: np.ndarray, masks: np.ndarray, crop_shape: Tuple[int]) -> A.Compose:
    """
    Do rigid augmentations and crop the patch to the size of the input_size.
    These are used after before patches are saved to hdf5 databases.
    The database access is several times faster if the patches are smaller.
    """
    transforms = compose([
        rigid_transforms(),
        center_crop(crop_shape[0], crop_shape[0])
    ])

    return transforms(image=image, masks=[masks])

