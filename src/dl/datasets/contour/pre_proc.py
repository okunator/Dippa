
import numpy as np
import skimage.morphology as morph
from typing import Tuple

from src.utils.mask_utils import center_crop

# Adapted from  https://github.com/vqdang/hover_net/blob/195ed9b6cc67b12f908285492796fb5c6c15a000/src/loader/augs.py#L21
def contours(inst_map: np.ndarray, crop_shape: Tuple[int]=(256, 256), thickness: int=1) -> np.ndarray:
    """
    Compute contours for every distinct nuclear object

    Args:
        inst_map (np.ndarray): 
            inst map
        crop_shape (Tndiuple[int]): 
            crop shape if network output smaller dims than the input
        thickness (int, default=2):
            thicnkness of the contour line. This specifies the disk 
            structuring element size.

    Returns:
        np.ndarray: distance maps of nuclei
    """

    if inst_map.shape[0] > crop_shape[0]: 
        inst_map = center_crop(inst_map, crop_shape[0], crop_shape[1])

    contour_map = np.zeros_like(inst_map, np.uint8)
    disk = morph.disk(thickness)
    inst_list = list(np.unique(inst_map))

    for inst_id in inst_list[1:]:
        inst = np.array(inst_map == inst_id, np.uint8)
        inner = morph.erosion(inst, disk)
        outer = morph.dilation(inst, disk)
        contour_map += outer - inner

    contour_map[contour_map > 0] = 1 # binarize

    return contour_map