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

from src.utils.process_utils import center_crop


def _augment(self, img, _):
    img = np.copy(img)
    orig_ann = img[...,0] # instance ID map
    fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
    crop_ann = center_crop(fixed_ann, crop_shape[0], crop_shape[1])

    # setting 1 boundary pix of each instance to background
    contour_map = np.zeros(fixed_ann.shape[:2], np.uint8)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0) # 0 is background

    k_disk = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], np.uint8)

    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inner = cv2.erode(inst_map, k_disk, iterations=1)
        outer = cv2.dilate(inst_map, k_disk, iterations=1)
        contour_map += outer - inner
    contour_map[contour_map > 0] = 1 # binarize
    img = np.dstack([fixed_ann, contour_map])
    return img