from src.dl.datasets.basic.dataset import BasicDataset
from src.dl.datasets.unet.dataset import UnetDataset
from src.dl.datasets.hover.dataset import HoverDataset
from src.dl.datasets.dist.dataset import DistDataset
from src.dl.datasets.contour.dataset import ContourDataset
from src.dl.datasets.augs import *


AUGS_LOOKUP = {
    "rigid":"rigid_transforms",
    "non_rigid":"non_rigid_transforms",
    "hue_sat":"hue_saturation_transforms",
    "blur":"blur_transforms",
    "non_spatial":"non_spatial_transforms",
    "random_crop":"random_crop",
    "center_crop":"center_crop",
    "resize":"resize",
    "normalize":"normalize"
}

DS_LOOKUP = {
    "basic":"BasicDataset",
    "hover":"HoverDataset",
    "dist":"DistDataset",
    "contour":"ContourDataset",
    "unet":"UnetDataset"
}