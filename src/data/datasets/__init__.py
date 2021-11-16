from .basic.dataset import BasicDataset
from .unet.dataset import UnetDataset
from .hover.dataset import HoverDataset
from .dist.dataset import DistDataset
from .contour.dataset import ContourDataset

from ._base._augs import (
    rigid_transforms, non_rigid_transforms, hue_saturation_transforms,
    blur_transforms, non_spatial_transforms, random_crop, center_crop,
    resize, normalize, to_tensor, compose, rigid_augs_and_crop
)


AUGS_LOOKUP = {
    "rigid": "rigid_transforms",
    "non_rigid": "non_rigid_transforms",
    "hue_sat": "hue_saturation_transforms",
    "blur": "blur_transforms",
    "non_spatial": "non_spatial_transforms",
    "random_crop": "random_crop",
    "center_crop": "center_crop",
    "resize": "resize",
    "normalize": "normalize"
}


DS_LOOKUP = {
    "basic": "BasicDataset",
    "hover": "HoverDataset",
    "dist": "DistDataset",
    "contour": "ContourDataset",
    "unet": "UnetDataset"
}


__all__ = [
    "AUGS_LOOKUP", "DS_LOOKUP", "rigid_transforms", "non_rigid_transforms",
    "hue_saturation_transforms", "blur_transforms", "non_spatial_transforms",
    "random_crop", "center_crop", "resize", "normalize", "ContourDataset",
    "DistDataset", "HoverDataset", "UnetDataset", "BasicDataset", "to_tensor",
    "compose", "rigid_augs_and_crop"
]
