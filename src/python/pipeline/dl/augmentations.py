import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms(patch_size=224):
    result = [
        albu.VerticalFlip(p=.5),
        albu.HorizontalFlip(p=.5),
        albu.HueSaturationValue(hue_shift_limit=(0,10), sat_shift_limit=0, val_shift_limit=0, p=1),
        albu.Rotate(p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
        albu.RandomSizedCrop((patch_size, patch_size), patch_size, patch_size),
        ToTensorV2()
    ]

    return result


def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
      albu.SmallestMaxSize(pre_size, p=1),
      albu.RandomCrop(image_size, image_size, p=1)
    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
      albu.LongestMaxSize(pre_size, p=1),
      albu.RandomCrop(image_size, image_size, p=1)
    ])

    # Converts the image to a square of size image_size x image_size
    result = [albu.OneOf([random_crop, rescale, random_crop_big], p=1)]
    return result
  

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensorV2()]


def no_transforms():
    # convert to torch.Tensor only
    return [ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([item for sublist in transforms_to_compose for item in sublist])
    return result