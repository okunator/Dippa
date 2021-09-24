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
import skimage.segmentation as segm
import scipy.ndimage as ndi

from ..cellpose.post_proc import enhance_hover
from src.utils.mask_utils import (
    remap_label,
    remove_debris,
    binarize,
    remove_small_objects
)


# Adapted from: https: // github.com/vqdang/hover_net/blob/master/src/postproc/hover.py 
def post_proc_hover(
        inst_map: np.ndarray,
        aux_map: np.ndarray,
        enhance: bool=True,
        **kwargs
    ) -> np.ndarray:
    """
    Post processing pipeline to combine hover branch output and instance
    segmentation branch output.

    Args:
    ----------
        inst_map (np.ndarray): 
            Soft inst map. Shape: (H, W, 2)
        aux_map (np.ndarray): 
            auxiliary map Shape: (H, W, 2). 
            aux_map[..., 0] = xmap, aux_map[..., 1] = ymap
        enhance (bool, default=True):
            Normalizes hover-maps to the 0-99 percentiles and clamps the
            values to min=-1 and max=1. like in CellPose. Results in 
            clearer signals.

    Returns:
    -----------
        np.ndarray: post-processed inst map. Shape (H, W)
    """

    inst_map = binarize(inst_map)
    h_dir = cv2.normalize(
        aux_map[..., 0], None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        aux_map[..., 1], None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    if enhance:
        enhanced_hover = enhance_hover(aux_map)
        h_dir = enhanced_hover[..., 0]
        v_dir = enhanced_hover[..., 1]

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (cv2.normalize(
        sobelh, None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ))
    sobelv = 1 - (cv2.normalize(
        sobelv, None, alpha=0, beta=1, 
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    )

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
    marker = remove_small_objects(marker, min_size=10)

    ws_temp = segm.watershed(dist, marker, mask=inst_map)

    inst_map = remove_debris(ws_temp, 18)
    inst_map = remap_label(inst_map)

    return inst_map 
