import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def pre_transforms(image_size=256):
    return [A.Resize(image_size, image_size, p=1)]


def test_transforms(input_size=256):
    result = [
        A.VerticalFlip(p=.5),
        A.HorizontalFlip(p=.5),
        A.HueSaturationValue(hue_shift_limit=(0,10), sat_shift_limit=0, val_shift_limit=0, p=1),
        A.Rotate(p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomSizedCrop((input_size, input_size), input_size, input_size),
        ToTensorV2()
    ]

    return result


def rigid_transforms(input_size=256):
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
    return result


def non_rigid_transforms():
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


def affine_transforms():
    return [A.ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
    )]


def hue_saturation_transforms():
    return [A.HueSaturationValue(
        hue_shift_limit=(0,15), sat_shift_limit=0, val_shift_limit=0, p=.5
    )]


def blur_transforms():
    return [
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.2)
    ]


def center_crop(input_size):
    return [
        A.CenterCrop(input_size, input_size, always_apply=True)
    ]


def random_crop(input_size):
    return [
        A.RandomCrop(input_size, input_size, always_apply=True)
    ]


def non_spatial_transforms():
    return [
        A.OneOf([
            A.CLAHE(p=0.8),
            A.RandomBrightnessContrast(p=0.8),    
            A.RandomGamma(p=0.8)
        ], p=0.2)
    ]


def to_tensor():
    return [ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = A.Compose([item for sublist in transforms_to_compose for item in sublist])
    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [A.Normalize(), ToTensorV2()]


def no_transforms():
    # convert to torch.Tensor only
    return [ToTensorV2()]
