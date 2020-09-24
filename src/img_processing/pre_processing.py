import cv2
import numpy as np
from scipy import ndimage as ndi

from scipy.ndimage.morphology import (
    distance_transform_cdt,
    distance_transform_edt,
    binary_dilation
)


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def fix_mirror_padding(ann):
    """
    Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.)
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0) # 0 is background
    fixed_ann = np.zeros(ann.shape[:2], np.int32)
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = ndi.label(inst_map)[0]
        remapped_ids[remapped_ids == 2] += current_max_id
        remapped_ids[remapped_ids == 1] = inst_id 
        fixed_ann[remapped_ids > 0] = remapped_ids[remapped_ids > 0]
        current_max_id = np.amax(fixed_ann)
    return fixed_ann


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def remove_1px_boundary(ann):
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
def get_weight_map(ann, inst_list, sigma=5.0, w0=10.0):
    if len(inst_list) <= 1: # 1 instance only
        return np.zeros(ann.shape[:2])
    stacked_inst_bgd_dst = np.zeros(ann.shape[:2] + (len(inst_list),))

    for idx, inst_id in enumerate(inst_list):
        inst_bgd_map = np.array(ann != inst_id , np.uint8)
        inst_bgd_dst = distance_transform_edt(inst_bgd_map)
        stacked_inst_bgd_dst[..., idx] = inst_bgd_dst

    near1_dst = np.amin(stacked_inst_bgd_dst, axis=2)
    near2_dst = np.expand_dims(near1_dst ,axis=2)
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
def gen_unet_labels(img, wc=None):
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


# ported from https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence
def rgb_clahe(in_rgb_img, grid_size = 8): 
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr[:,:,[2,1,0]]


# ported from https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence
def rgb_clahe_justl(in_rgb_img, grid_size = 8): 
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size,grid_size))
    return clahe.apply(lab[:,:,0])
