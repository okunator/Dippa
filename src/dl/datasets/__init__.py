from src.dl.datasets.basic.dataset import BasicDataset
from src.dl.datasets.unet.dataset import UnetDataset
from src.dl.datasets.hover.dataset import HoverDataset
from src.dl.datasets.micro.dataset import MicroDataset

AUGS_LOOKUP = {
    "rigid":"rigid_transforms",
    "non_rigid":"non_rigid_transforms",
    "affine":"affine_transforms",
    "hue_sat":"hue_saturation_transforms",
    "blur":"blur_transforms",
    "non_spatial":"non_spatial_transforms",
    "random_crop":"random_crop",
    "center_crop":"center_crop",
}

DS_LOOKUP = {
    "basic":"BasicDataset",
    "hover":"HoverDataset",
    "micro":"MicroDataset",
    "unet":"UnetDataset"
}