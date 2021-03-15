"""
MIT License

Copyright (c) 2018 PeterJackNaylor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import skimage.segmentation as segm
import skimage.measure as measure
import skimage.morphology as morph
import scipy.ndimage as ndi
from skimage import img_as_ubyte

from src.utils.mask_utils import binarize
from src.utils.img_utils import percentile_normalize_and_clamp


# Adapted from https://github.com/PeterJackNaylor/DRFNS/blob/master/src_RealData/postproc/postprocessing.py
def h_minima_reconstruction(inv_dist_map: np.ndarray, lamb: int=7) -> np.ndarray:
    """
    Performs a H minimma reconstruction via an erosion method.

    Args:
    ----------
        inv_dist_map (np.ndarray):
            inverse distance map
        lamb (int, default=7):
            intensity shift value

    Returns:
    ----------
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
    ----------
        inv_dist_map (np.ndarray):
            Inverse distance map. Shape (H, W)
        mask (np.ndarray, default=None):
            binary mask to remove small debris. Shape (H, W)

    Returns:
    ----------
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
    ----------
        dist_map (np.ndarray):
            The regressed distance transform from auxiliary branch of the network
        binary_mask (np.ndarray):
            The thresholded probability map from the binary seg branch of the network
        thresh (float, default=0.5):
            The threshold value to find markers from the dist_map

    Returns:
    ----------
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


def post_proc_drfns(dist_map: np.ndarray, inst_map: np.ndarray, thresh: float=0.5) -> np.ndarray:
    """
    Post processing pipeline introduced in:
    https://ieeexplore.ieee.org/document/8438559

    Slightly modified. Uses the thresholded prob_map as the mask param in watershed.
    Markers are computed from the regressed distance map (inverted).

    Uses regressed distance transform from an auxiliary branch to separate clumped nuclei. 

    Args:
    -----------
        dist_map (np.ndarray):
            Regressed distance transform from the auxiliary branch. Shape (H, W, 1)
        inst_map (np.ndarray):
            The segmentation mask from the binary seg branch of thenetwork. Shape (H, W)
            If inst_map is labelled it will be binarized.
        thresh (float, default=0.5):
            threshold value for markers and binary mask 

    Returns:
    -----------
            np.ndarray, the post-processed inst_map. Same shape as input (H, W)
    """
    dist_map = percentile_normalize_and_clamp(dist_map, a_min=0, a_max=1)
    binary_mask = binarize(inst_map)
    result = dynamic_ws_alias(dist_map, binary_mask, thresh)

    return result