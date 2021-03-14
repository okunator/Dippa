import cv2
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph

from src.utils.mask_utils import binarize
from src.utils.img_utils import percentile_normalize_and_clamp
from ..thresholding import naive_thresh_prob


# Adapted from https://github.com/vqdang/hover_net/blob/tensorflow-final/src/postproc/other.py
def post_proc_dcan(prob_map: np.ndarray, contour_map: np.ndarray) -> np.ndarray:
    """
    DCAN post-processing pipeline:
    https://arxiv.org/abs/1604.02677

    Args:
    ----------
        prob_map (np.ndarray):
            Probablilty map of the nuclei. Shape (H, W)
        contour_map (np.ndarray):
            Prediction from the contour branch of the network. Shape (H, W)
    """
    contour_map = percentile_normalize_and_clamp(contour_map)
    sub = prob_map - contour_map
    pre_insts = naive_thresh_prob(sub)
    
    inst_ids = np.unique(pre_insts)[1:]
    disk = morph.disk(3)
    inst_map = np.zeros_like(pre_insts)
    for inst_id in inst_ids:
        inst = np.array(pre_insts == inst_id, dtype=np.uint8)
        inst = cv2.dilate(inst, disk, iterations=1)
        inst = ndi.binary_fill_holes(inst)
        inst_map[inst > 0] = inst_id

    return inst_map