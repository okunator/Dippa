"""
MIT License

Copyright (c) 2020 vqdang

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

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology as morph
from scipy.ndimage.morphology import distance_transform_edt

# These are used for every dataset

# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def fix_mirror_padding(inst_map: np.ndarray) -> np.ndarray:
    """
    Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.)

    Args:
        inst_map (np.ndarray): inst map
    """
    current_max_id = np.amax(inst_map)
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0) # 0 is background
    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        remapped_ids = ndi.label(inst)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        inst_map[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(inst_map)

    return inst_map


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
# minor tweaks
def remove_1px_boundary(inst_map: np.ndarray) -> np.ndarray:
    """
    Removes 1px around each instance, removing overlaps of cells in an inst map

    Args: 
        inst_map (np.ndarray): inst map
    """
    new_inst_map = np.zeros(inst_map.shape[:2], np.int32)
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0) # 0 is background
    k = morph.disk(1)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        inst = cv2.erode(inst, k, iterations=1)
        new_inst_map[inst > 0] = inst_id
    return new_inst_map


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
# minor tweaks
def get_weight_map(inst_map: np.ndarray, sigma: float = 5.0, w0: float = 10.0):
    """
    Generate a weight map like in U-Net paper

    Args: 
        inst_map (np.ndarray): Instance map
        sigma (float): factor multiplied to the for the distance maps
        w0 (float): weight multiplied to the penalty map 
    """
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0) # 0 is background

    if len(inst_list) <= 1: # 1 instance only
        return np.zeros(inst_map.shape[:2])
    stacked_inst_bgd_dst = np.zeros(inst_map.shape[:2] + (len(inst_list),))

    for idx, inst_id in enumerate(inst_list):
        inst_bgd_map = np.array(inst_map != inst_id , np.uint8)
        inst_bgd_dst = distance_transform_edt(inst_bgd_map)
        stacked_inst_bgd_dst[..., idx] = inst_bgd_dst

    near1_dst = np.amin(stacked_inst_bgd_dst, axis=2)
    near2_dst = np.expand_dims(near1_dst, axis=2)
    near2_dst = stacked_inst_bgd_dst - near2_dst
    near2_dst[near2_dst == 0] = np.PINF # very large
    near2_dst = np.amin(near2_dst, axis=2)
    near2_dst[inst_map > 0] = 0 # the instances
    near2_dst = near2_dst + near1_dst
    # to fix pixel where near1 == near2
    near2_eve = np.expand_dims(near1_dst, axis=2)
    # to avoide the warning of a / 0
    near2_eve = (1.0 + stacked_inst_bgd_dst) / (1.0 + near2_eve)
    near2_eve[near2_eve != 1] = 0
    near2_eve = np.sum(near2_eve, axis=2)
    near2_dst[near2_eve > 1] = near1_dst[near2_eve > 1]
    #
    pix_dst = near1_dst + near2_dst
    pen_map = pix_dst / sigma
    pen_map = w0 * np.exp(- pen_map**2 / 2)
    pen_map[inst_map > 0] = 0 # inner instances zero
    return pen_map


def binarize(inst_map: np.ndarray) -> np.ndarray:
    """
    Binarize a labelled instance map

    Args:
        inst_map (np.ndarray): instance map to be binarized
    """
    inst_map[inst_map > 0] = 1
    return inst_map.astype("uint8")


# From https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/misc.py
# warning removed
def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False, *, out=None):
    """Remove objects smaller than the specified size.
    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.
    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    in_place : bool, optional (default: False)
        If ``True``, remove the objects in the input array itself.
        Otherwise, make a copy. Deprecated since version 0.19. Please
        use `out` instead.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.
    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.
    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.
    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True
    """

    if out is not None:
        in_place = False

    if in_place:
        out = ar
    elif out is None:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

