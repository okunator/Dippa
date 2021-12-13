"""
Copyright © 2020 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer. Redistributions in binary 
form must reproduce the above copyright notice, this list of conditions and 
the following disclaimer in the documentation and/or other materials provided
with the distribution. Neither the name of HHMI nor the names of its contributors
may be used to endorse or promote products derived from this software without 
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
"""

# imported from the cellpose repo. Code style mods for (my own) 
# readability.

import numpy as np

from skimage.color import hsv2rgb
from scipy.ndimage import find_objects, binary_fill_holes

from src.utils import (
    percentile_normalize, percentile_normalize_and_clamp
)


__all__ = [
    "normalize99", "enhance_hover", "flows_from_hover", 
    "fill_holes_and_remove_small_masks"
]


def normalize99(
        Y: np.ndarray,
        lower: float=0.01,
        upper: float=99.99
    ) -> np.ndarray:
    """
    Normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th 
    percentile
    
    Args:
    --------
        Y (np.ndarray):
            Input img.
        lower (float, default=0.01):
            The lower percentile
        upper (float, default=99.99):
            The upper percentile
            
    Returns:
    --------
        np.ndarray. Normalized img. Same shape as input.
    """
    X = Y.copy()
    
    return np.interp(
        X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1)
    )


def enhance_hover(
        hover: np.ndarray,
        order: str="XY",
        channels: str="HWC"
    ) -> np.ndarray:
    """
    Normalizes to the 0-99 percentiles and clamps the values of the 
    horizontal & vertical maps like in CellPose. Assumes that the 
    hovermaps are stacked in "YX" order. And that the shape = (H, W, 2).
    If not, then reshapes the np.ndarray.

    This is ported from the CellPose repo and slightly modified.

    Args:
    -----------
        hover (np.ndarray):
            Regressed Horizontal and Vertical gradient maps. 
            Shape (2, H, W)|(H, W, 2)
        order (str, default="XY"):
            The order the horizontal and vertical maps are stacked. 
            Horizontal = "X", vertical = "Y". In Cellpose the order is 
            "YX". Networks in this repo output "XY"
        channels (str, default="HWC"):
            The order of image dimensions. One of :"HW", "HWC", "CHW"
    
    Returns:
    -----------
        np.ndarray: Enhanced hover-maps with channel order "YX" and 
        shape (H, W, 2)
    """
    if order not in ("XY", "YX"):
        raise ValueError(f"""
            `order` not in {("XY", "YX")}. Got: {order}"""
        )
    if channels not in ("HWC", "CHW"):
        raise ValueError(f"""
            `channels` not in {("HWC", "CHW")}. Got: {channels}"""
        )

    if channels == "CHW":
        hover = hover.transpose(1, 2, 0) # HWC

    if order == "YX":
        hover = hover[..., (1, 0)] #YX

    enhanced = percentile_normalize_and_clamp(hover, a_min=-1, a_max=1)
    return enhanced


def flows_from_hover(
        hover: np.ndarray,
        order: str="XY",
        channels: str="HWC"
    ) -> np.ndarray:
    """
    Convert Horizontal and Vertical gradients to gradient flows like in 
    CellPose. 
    
    This is ported from the CellPose repo and slightly modified.

    Args:
    ----------
        hover (np.ndarray):
            Regressed Horizontal and Vertical gradient maps. Shape (H, W, 2).
            order is "XY". i.e. Horizontal map in hover[..., 0] and vertical
            in hover[..., 1]
        order (str, default="XY"):
            The order the horizontal and vertical maps are stacked. Horizontal = "X"
            Vertical = "Y". In Cellpose the order is "YX". Networks in this repo output "XY"
        channels (str, default="HWC"):
            The order of image dimensions. One of ("HW", "HWC", "CHW")

    Returns:
    ----------
        np.ndarray: the optical flow representation. Shape (H, W, 3)
    """
    enhanced = enhance_hover(hover, order=order, channels=channels)
    H = (np.arctan2(enhanced[..., 1], enhanced[..., 0]) + np.pi) / (2*np.pi)
    S = percentile_normalize(enhanced[..., 1]**2 + enhanced[..., 0]**2, channels="HW")
    HSV = np.stack([H, S, S], axis=-1)
    HSV = np.clip(HSV, a_min=0.0, a_max=1.0)
    flow = (hsv2rgb(HSV)*255).astype(np.uint8)

    return flow


def fill_holes_and_remove_small_masks(
        masks: np.ndarray,
        min_size: int=15
    ) -> np.ndarray:
    """ 
    fill holes in masks 2D and discard masks smaller than min_size
    
    Parameters
    ----------------
    masks: int, 2D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx]
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------
    masks: int, 2D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx]
    
    """
    if masks.ndim != 2:
        raise ValueError(
            'masks_to_outlines takes 2D, not %dD array'%masks.ndim
        )
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:
                msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1

    return masks
