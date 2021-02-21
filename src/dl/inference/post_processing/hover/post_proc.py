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
import skimage.morphology as morph
import skimage.feature as feat
import skimage.segmentation as segm
import skimage.filters as filters
import skimage.util as util
import scipy.ndimage as ndi

from src.utils.process_utils import bounding_box, remap_label
from ..utils import remove_debris, binarize


# ported from: https: // github.com/vqdang/hover_net/blob/master/src/postproc/hover.py  # L69
def post_proc_hover(inst_map: np.ndarray, aux_map: np.ndarray, **kwargs) -> np.ndarray:
    """
    Post processing pipeline to combine hover branch output and instance segmentation branch output.

    Args:
        inst_map (np.ndarray): 
            Soft inst map. Shape: (H, W, 2)
        aux_map (np.ndarray): 
            Shape: (H, W, 2). aux_map[..., 0] = xmap, aux_map[..., 1] = ymap

    Returns:
        np.ndarray that is processed in the same way as in github.com/vqdang/hover_net/blob/master/src/postproc/hover.py
    """
    inst_map = binarize(inst_map)

    h_dir = cv2.normalize(aux_map[..., 0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(aux_map[..., 1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - inst_map)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * inst_map
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall[overall >= 0.4] = 1
    overall[overall <  0.4] = 0
    
    marker = inst_map - overall
    marker[marker < 0] = 0
    marker = ndi.binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = ndi.label(marker)[0]
    marker = morph.remove_small_objects(marker, min_size=10)

    ws_temp = segm.watershed(dist, marker, mask=inst_map)

    inst_map = remove_debris(ws_temp, 18)
    inst_map = remap_label(inst_map)

    return inst_map 


# Treid to do some mods to the original. Not as quite good...
def post_proc_hover2(aux_map: np.ndarray, inst_map: np.ndarray, sigma: float = 2.0, **kwargs):
    """
    Post processing pipeline to combine hover branch output and instance segmentation branch output.

    Args:
        inst_map (np.ndarray):  inst map. Shape: (H, W, 2)
        aux_map (np.ndarray): Shape: (H, W, 2). hover_maps[..., 0] = xmap, hover_maps[..., 1] = ymap

    Returns:
        np.ndarray that is processed similarly as in github.com/vqdang/hover_net/blob/master/src/postproc/hover.py
        and then a little further
    """

    hdir = cv2.normalize(aux_map[..., 0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    vdir = cv2.normalize(aux_map[..., 1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(hdir, cv2.CV_64F, 1, 0, ksize=25)
    sobelv = cv2.Sobel(vdir, cv2.CV_64F, 0, 1, ksize=25)

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    maxed = np.maximum(sobelh, sobelv)
    dist = (1.0 - maxed) * inst_map
    dist = cv2.GaussianBlur(dist, (3, 3), 0)

    maxed_shape = np.nan_to_num(feat.shape_index(1-maxed, sigma=sigma))
    maxed_shape *= inst_map
    maxed_shape[maxed_shape <= 1/8] = 0
    maxed_shape[maxed_shape != 0] = 1
    maxed_shape = ndi.binary_fill_holes(maxed_shape.astype(bool)).astype('uint8')
    maxed_shape = morph.remove_small_objects(maxed_shape.astype(bool), 8, connectivity=1)
    maxed_shape = ndi.label(maxed_shape)[0]

    markers = maxed_shape
    ws_temp = segm.watershed(-dist, mask=inst_map, markers=markers, watershed_line=True)
    ws_temp = ndi.label(ws_temp)[0]

    id_count = 1
    cell_ids = np.unique(ws_temp)[1:]
    mask_new = np.zeros(ws_temp.shape[:2], dtype=np.int32)
    for nuc_id in cell_ids:
        nuc_map = np.copy(ws_temp == nuc_id)
        y1, y2, x1, x2 = bounding_box(nuc_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        nuc_map_crop = nuc_map[y1:y2, x1:x2].astype("int32")

        # Gets rid of small stuff
        nuc_map_crop = filters.rank.median(util.img_as_ubyte(nuc_map_crop), morph.disk(3))

        # Fill holes
        nuc_map_crop = ndi.binary_fill_holes(nuc_map_crop)

        mask_inst = nuc_map_crop*nuc_id
        mask_new[y1:y2, x1:x2] += mask_inst

        # if cells end up overlapping after dilations then remove the overlaps
        # so no new ids are created when summing overlapping ids to the result mask
        new_ids = np.unique(mask_new)[1:]
        if id_count < len(new_ids):
            for ix in new_ids[int(np.where(new_ids == nuc_id)[0]+1):]:
                mask_new[mask_new == ix] = 0
        id_count += 1

    inst_map = remove_debris(mask_new, 18)
    inst_map = remap_label(inst_map)

    return inst_map