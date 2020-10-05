import cv2
import numpy as np
import albumentations as A
import ttach as tta
from typing import List, Any
from albumentations.pytorch import ToTensorV2


def pre_transforms(image_size: int = 256) -> List[Any]:
    return [A.Resize(image_size, image_size, p=1)]


def test_transforms(input_size: int = 256) -> List[Any]:
    result = [
        A.VerticalFlip(p=.5),
        A.HorizontalFlip(p=.5),
        A.HueSaturationValue(hue_shift_limit=(0,10), sat_shift_limit=0, val_shift_limit=0, p=1),
        A.Rotate(p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomSizedCrop((input_size, input_size), input_size, input_size),
        ToTensorV2()
    ]

    return result


def rigid_transforms(input_size: int = 256) -> List[Any]:
    return [
        A.OneOf([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
            )
        ])
    ]


def non_rigid_transforms() -> List[Any]:
    return [
        A.OneOf([
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3
            ),
            A.GridDistortion(p=0.6),
            A.OpticalDistortion(
                distort_limit=2, shift_limit=0.5, p=0.2
            )
        ], p=0.25)
    ]


def affine_transforms() -> List[Any]:
    return [A.ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
    )]


def hue_saturation_transforms() -> List[Any]:
    return [A.HueSaturationValue(
        hue_shift_limit=(0,15), sat_shift_limit=0, val_shift_limit=0, p=.5
    )]


def blur_transforms() -> List[Any]:
    return [
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.2)
    ]


def center_crop(input_size: int) -> List[Any]:
    return [
        A.CenterCrop(input_size, input_size, always_apply=True, p=1)
    ]


def random_crop(input_size: int) -> List[Any]:
    return [
        A.RandomCrop(input_size, input_size, always_apply=True, p=1)
    ]


def non_spatial_transforms() -> List[Any]:
    return [
        A.OneOf([
            A.CLAHE(p=0.8),
            A.RandomBrightnessContrast(p=0.8),    
            A.RandomGamma(p=0.8)
        ], p=0.2)
    ]


def to_tensor() -> List[Any]:
    return [ToTensorV2()]


def compose(transforms_to_compose: List[Any]) -> A.Compose:
    # combine all augmentations into one single pipeline
    result = A.Compose([item for sublist in transforms_to_compose for item in sublist])
    return result


def no_transforms() -> List[Any]:
    # convert to torch.Tensor only
    return [ToTensorV2()]


# ttach transformations (turns out these take too much memory)
def tta_transforms() -> tta.Compose:
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),      
        ]
    )


def tta_augs() -> List[Any]:
    return [
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1),
        A.Rotate((45, 45), p=1),
        A.Rotate((95, 95), p=1),
        A.Rotate((155, 155), p=1),
        A.Rotate((245, 245), p=1),
    ]


def tta_deaugs() -> List[Any]:
    return [
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1),
        A.Rotate((-45, -45), p=1),
        A.Rotate((-95, -95), p=1),
        A.Rotate((-155, -155), p=1),
        A.Rotate((-245, -245), p=1),
    ]


def tta_five_crops(io: np.ndarray) -> List[Any]:
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


def resize(height: int, width: int) -> A.Resize:
    return A.Resize(height, width)


def rigid_augs_and_crop(image: np.ndarray, mask: np.ndarray, input_size: int) -> A.Compose:
    """
    Do rigid augmentations and crop the patch to the size of the input_size.
    These are used after before patches are saved to hdf5 databases.
    The database access is several times faster if the patches are smaller.
    """
    transforms = compose([
        rigid_transforms(),
        #random_crop(input_size),
        center_crop(input_size)
    ])

    return transforms(image=image, mask=mask)
