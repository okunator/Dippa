import cv2
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
import skimage.segmentation as segm

from src.utils.mask_utils import binarize, remove_small_objects
from src.utils.img_utils import percentile_normalize_and_clamp
from ..thresholding import naive_thresh_prob


def post_proc_dran(prob_map: np.ndarray, contour_map: np.ndarray) -> np.ndarray:
    """
    DRAN post-processing pipeline:
    https://www.frontiersin.org/articles/10.3389/fbioe.2019.00053/full

    This is not the original implementation but follows along the steps
    introduced in the paper. Added dilation to the end of the pipeline.

    Args:
    ----------
        prob_map (np.ndarray):
            Probablilty map of the nuclei. Shape (H, W)
        contour_map (np.ndarray):
            Prediction from the contour branch of the network. Shape (H, W)

    Returns:
    ----------
        np.ndarray: post-processed inst map
    """
    # Thresh the contour map and dilate
    contour = percentile_normalize_and_clamp(contour_map, a_min=0, a_max=1)
    cnt_binary = binarize(naive_thresh_prob(contour))
    cnt_binary = cv2.dilate(cnt_binary, morph.disk(3), iterations=1)

    # thresh the binary map and remove artefacts
    binary = binarize(naive_thresh_prob(prob_map))
    binary = remove_small_objects(binary.astype(bool), min_size=10).astype("uint8")

    # subtract contour map from binary and use as markers 
    markers = (binary - cnt_binary)
    markers[markers != 1] = 0
    markers = ndi.label(markers)[0]

    # Find the distance map of the markers and normalize
    distance = ndi.distance_transform_edt(markers)
    distance = 255 * (distance / np.amax(distance))

    # watershed
    inst_map = segm.watershed(-distance, markers=markers, mask=binary)

    return inst_map