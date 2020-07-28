import cv2
import numpy as np
from skimage.morphology import opening 
from scipy import ndimage as ndi
from scipy.ndimage.morphology import (distance_transform_cdt,
                                      distance_transform_edt,
                                      binary_dilation)


# ported from https://github.com/vqdang/hover_net/blob/master/src/misc/utils.py
def cropping_center(x, crop_shape, batch=False):   
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:,h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]        
    return x


# ported from https://github.com/vqdang/hover_net/blob/master/src/misc/viz_utils.py
def bounding_box(img):
    """
    Bounding box coordinates for nuclei instance
    that is given as input.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out 
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


# ported from https://github.com/nicolefinnie/kaggle-dsb2018/blob/master/src/modules/image_processing.py
def mask_contours(mask):
    mask = mask.astype(np.uint8)*255
    
    # First find contours for the binary mask
    # Pad first, b/c cv2 computes contours starting from 1st pixel
    # (borders wont get straight contours)
    bg = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    bg_pad = np.pad(bg.copy(), ((2, 2), (2, 2)), 'edge')
    bg_rgb = cv2.cvtColor(bg_pad, cv2.COLOR_GRAY2RGB)
    mask_pad = np.pad(mask.copy(), ((2 ,2), (2, 2)), 'edge')
    _, thrsh = cv2.threshold(mask_pad, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thrsh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contoured_rgb = cv2.drawContours(
        bg_rgb, contours, contourIdx=-1, 
        color=(255,255,255), thickness=1
    )
    
    contoured_gray = cv2.cvtColor(contoured_rgb, cv2.COLOR_RGB2GRAY)
    contoured_mask = contoured_gray[2:-2, 2:-2]
            
    return contoured_mask


def instance_contours(label_img, thickness=2):
    """Find a contour for each nuclei instance in a mask"""
    
    # Padding first to avoid contouring the image borders
    label_img2 = np.pad(label_img.copy(), ((thickness, thickness), (thickness, thickness)), 'edge')
    bg = np.zeros(label_img2.shape, dtype=np.uint8)
    for j, nuc_id in enumerate(np.unique(label_img)):
        inst_map = np.array(label_img2 == nuc_id, np.uint8)
        y1, y2, x1, x2  = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= label_img.shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= label_img.shape[0] - 1 else y2
                
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_bg_crop = bg[y1:y2, x1:x2]
        contours, hierarchy = cv2.findContours(inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contoured_rgb = cv2.drawContours(
            inst_bg_crop, contours, contourIdx=-1, 
            color=(255, 255, 255), thickness=thickness
        )
        
        bg[y1:y2, x1:x2] = inst_bg_crop
    
    bg = bg[thickness:-thickness, thickness:-thickness]
    return bg
