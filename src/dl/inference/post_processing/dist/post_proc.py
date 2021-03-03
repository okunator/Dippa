import numpy as np
import skimage.segmentation as segm
import skimage.measure as measure
import skimage.morphology as morph
import scipy.ndimage as ndi
from skimage import img_as_ubyte

from src.utils.mask_utils import binarize


def enhance_dist(dist: np.ndarray, channels: str="HWC") -> np.ndarray:
    """
    Normalizes to the 0-99 percentiles and clamps the values of the dist map.
    Same normalization/enhancing also in cellpose and hover-net

    Args:
        hover (np.ndarray):
            Regressed Horizontal and Vertical gradient maps. Shape (1, H, W)|(H, W, 1)
        channels (str, default="HWC"):
            The order of image dimensions. One of ("HW", "HWC", "CHW")
    
    Returns:
        Enhanced dist-map (np.ndarray) with shape (H, W)
    """
    assert channels in ("HWC", "CHW")

    if channels == "CHW":
        dist = dist.transpose(1, 2, 0) # HWC

    percentile99 = np.percentile(dist, q=99, axis=(0, 1))
    percentile1 = np.percentile(dist, q=1, axis=(0, 1))
    percentiles = np.stack([percentile99, percentile1])
    colmax = np.max(percentiles, axis=0)
    enhanced = np.clip((dist / colmax), a_min=0, a_max=1)

    return enhanced


def h_minima_reconstruction(inv_dist_map: np.ndarray, lamb: int=7) -> np.ndarray:
    """
    Performs a H minimma reconstruction via an erosion method.

    Args:
        inv_dist_map (np.ndarray):
            inverse distance map
        lamb (int, default=7):
            intensity shift value

    Returns:
        np.ndarray. H minima reconstruction from the inverse distance transform. Shape (H, W)
    """

    def making_top_mask(x: np.ndarray, lamb: int=lamb) -> int:
        return min(255, x + lamb)

    # vectorize for performance
    find_minima = np.vectorize(making_top_mask)
    shift_inv_dist_map = find_minima(inv_dist_map)

    # reconstruct
    seed = shift_inv_dist_map
    mask = inv_dist_map
    reconstructed = morph.reconstruction(seed, mask, method="erosion").astype("uint8")
    return reconstructed


def find_maxima(inv_dist_map: np.ndarray, mask: np.ndarray=None) -> np.ndarray:
    """
    Finds all local maxima from 2D image.

    Args:
        inv_dist_map (np.ndarray):
            Inverse distance map. Shape (H, W)
        mask (np.ndarray, default=None):
            binary mask to remove small debris. Shape (H, W)

    Returns:
        np.ndarray, the found maxima. Shape (H, W).

    """
    reconstructed = h_minima_reconstruction(inv_dist_map, 40)

    res = reconstructed - inv_dist_map
    if mask is not None:
        res[mask==0] = 0

    return res


def dynamic_ws_alias(dist_map: np.ndarray, binary_mask: np.ndarray, thresh: float=0.5) -> np.ndarray:
    """
    Runs the dynamic watershed segmentation. 
    
    Removed the suspicious stuff from the end. Just obscured the results...

    Args:
        dist_map (np.ndarray):
            The regressed distance transform from auxiliary branch of the network
        binary_mask (np.ndarray):
            The thresholded probability map from the binary seg branch of the network
        thresh (float, default=0.5):
            The threshold value to find markers from the dist_map

    Returns:
        np.ndarray, the labelled watershed segmentation result. Shape (H, W)
    """
    # binarize probs and dist map
    binary_dist_map = (dist_map > thresh)

    # invert distmap
    inv_dist_map = 255 - img_as_ubyte(dist_map)

    # find markers from minima erosion reconstructed maxima of inv dist map
    reconstructed = h_minima_reconstruction(inv_dist_map)
    markers = find_maxima(reconstructed, mask=binary_dist_map)
    markers = ndi.label(markers)[0]

    # apply watershed
    ws = segm.watershed(reconstructed, markers, mask=binary_mask)

    return ws


def post_proc_dist(dist_map: np.ndarray, inst_map: np.ndarray, thresh: float=0.5) -> np.ndarray:
    """
    Post processing pipeline introduced in:
    https://ieeexplore.ieee.org/document/8438559

    Slightly modified. Uses the thresholded prob_map as the mask param in watershed.
    Markers are computed from the regressed distance map (inverted).

    Uses regressed distance transform from an auxiliary branch to separate clumped nuclei. 

    Args:
        dist_map (np.ndarray):
            Regressed distance transform from the auxiliary branch. Shape (H, W, 1)
        inst_map (np.ndarray):
            The segmentation mask from the binary seg branch of thenetwork. Shape (H, W)
            If inst_map is labelled it will be binarized.
        thresh (float, default=0.5):
            threshold value for markers and binary mask 
    """
    dist_map = enhance_dist(dist_map.squeeze()) 
    binary_mask = binarize(inst_map)
    result = dynamic_ws_alias(dist_map, binary_mask, thresh)

    return result