import cv2
import numpy as np
from scipy import ndimage as ndi
from typing import Optional, List, Tuple
from skimage import morphology as morph

from scipy.ndimage.morphology import (
    distance_transform_cdt,
    distance_transform_edt,
    binary_dilation
)

from src.img_processing.process_utils import (
    bounding_box, cropping_center
)

# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def fix_mirror_padding(ann: np.ndarray) -> np.ndarray:
    """
    Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.)
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0) # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = ndi.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]

        current_max_id = np.amax(ann)
    return ann


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def remove_1px_boundary(ann: np.ndarray) -> np.ndarray:
    new_ann = np.zeros(ann.shape[:2], np.int32)
    inst_list = list(np.unique(ann))
    inst_list.remove(0) # 0 is background

    k = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], np.uint8)

    for idx, inst_id in enumerate(inst_list):
        inst_map = np.array(ann == inst_id, np.uint8)
        inst_map = cv2.erode(inst_map, k, iterations=1)
        new_ann[inst_map > 0] = inst_id
    return new_ann


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def get_weight_map(ann: np.ndarray, inst_list: List[int], sigma: float = 5.0, w0: float = 10.0):
    if len(inst_list) <= 1: # 1 instance only
        return np.zeros(ann.shape[:2])
    stacked_inst_bgd_dst = np.zeros(ann.shape[:2] + (len(inst_list),))

    for idx, inst_id in enumerate(inst_list):
        inst_bgd_map = np.array(ann != inst_id , np.uint8)
        inst_bgd_dst = distance_transform_edt(inst_bgd_map)
        stacked_inst_bgd_dst[..., idx] = inst_bgd_dst

    near1_dst = np.amin(stacked_inst_bgd_dst, axis=2)
    near2_dst = np.expand_dims(near1_dst, axis=2)
    near2_dst = stacked_inst_bgd_dst - near2_dst
    near2_dst[near2_dst == 0] = np.PINF # very large
    near2_dst = np.amin(near2_dst, axis=2)
    near2_dst[ann > 0] = 0 # the instances
    near2_dst = near2_dst + near1_dst
    # to fix pixel where near1 == near2
    near2_eve = np.expand_dims(near1_dst ,axis=2)
    # to avoide the warning of a / 0
    near2_eve = (1.0 + stacked_inst_bgd_dst) / (1.0 + near2_eve)
    near2_eve[near2_eve != 1] = 0
    near2_eve = np.sum(near2_eve, axis=2)
    near2_dst[near2_eve > 1] = near1_dst[near2_eve > 1]
    #
    pix_dst = near1_dst + near2_dst
    pen_map = pix_dst / sigma
    pen_map = w0 * np.exp(- pen_map**2 / 2)
    pen_map[ann > 0] = 0 # inner instances zero
    return pen_map


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def gen_unet_labels(img: np.ndarray, wc: Optional[float] = None):
    img = np.copy(img)
    fixed_ann = fix_mirror_padding(img)
    
    # setting 1 boundary pix of each instance to background
    fixed_ann = remove_1px_boundary(fixed_ann)

    # cant do the shortcut because near2 also needs instances 
    # outside of cropped portion
    inst_list = list(np.unique(fixed_ann))
    inst_list.remove(0) # 0 is background
    wmap = get_weight_map(fixed_ann, inst_list)

    if wc is None:             
        wmap += 1 # uniform weight for all classes
    else:
        class_weights = np.zeros_like(fixed_ann.shape[:2])
        for class_id, class_w in wc.items():
            class_weights[fixed_ann == class_id] = class_w
        wmap += class_weights

    # fix other maps to align
    img[fixed_ann == 0] = 0 

    return img, wmap


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def gen_hv_maps(inst_map: np.ndarray, crop_shape: Tuple[int] = (256, 256)) -> np.ndarray:
    """
        Input annotation must be of original shape.
        
        The map is calculated only for instances within the crop portion
        but based on the original shape in original image.
    
        Perform following operation:
        Obtain the horizontal and vertical distance maps for each
        nuclear instance.
    """
    fixed_ann = fix_mirror_padding(inst_map)

    # re-cropping with fixed instance id map
    # (if output from network smaller than input)
    crop_ann = cropping_center(fixed_ann, crop_shape)
    # crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(inst_map.shape[:2], dtype=np.float32)
    y_map = np.zeros(inst_map.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst = np.array(crop_ann == inst_id, np.int32)
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
        "ymap":y_map,
        "xmap":x_map
    }


