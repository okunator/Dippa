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
from typing import Tuple
from skimage import morphology as morph

from src.utils.process_utils import bounding_box, center_crop


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def gen_hv_maps(inst_map: np.ndarray, crop_shape: Tuple[int] = (256, 256)) -> np.ndarray:
    """
    Generates horizontal and vertical maps from instance labels

    Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    Args:
        inst_map (np.ndarray): inst map
        crop_shape (Tuple[int]): crop shape if network output smaller dims than the input
    """

    crop_inst_map = center_crop(inst_map, crop_shape[0], crop_shape[1])
    crop_inst_map = morph.remove_small_objects(crop_inst_map, min_size=30)

    x_map = np.zeros(inst_map.shape[:2], dtype=np.float32)
    y_map = np.zeros(inst_map.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_inst_map))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst = np.array(crop_inst_map == inst_id, np.int32)
        y1, y2, x1, x2 = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst = inst[y1:y2, x1:x2]

        # instance center of mass, rounded to nearest pixel
        inst_com = list(ndi.measurements.center_of_mass(inst))
        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst.shape[1]+1)
        inst_y_range = np.arange(1, inst.shape[0]+1)

        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst == 0] = 0
        inst_y[inst == 0] = 0
        inst_x = inst_x.astype('float32')
        inst_y = inst_y.astype('float32')

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= (-np.amin(inst_x[inst_x < 0]))
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= (-np.amin(inst_y[inst_y < 0]))
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= (np.amax(inst_x[inst_x > 0]))
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= (np.amax(inst_y[inst_y > 0]))

        ####
        x_map_box = x_map[y1:y2, x1:x2]
        x_map_box[inst > 0] = inst_x[inst > 0]

        y_map_box = y_map[y1:y2, x1:x2]
        y_map_box[inst > 0] = inst_y[inst > 0]

    return {
        "xmap":x_map,
        "ymap":y_map
    }