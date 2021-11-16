import cv2
import numpy as np
from typing import Dict, Optional

from .mask_utils import bounding_box


KEY_COLORS = {
    "background": (255., 255., 255.),
    "miscellanous": (0., 200., 100.),
    "inflammatory": (165., 255., 255.),
    "epithelial": (255., 255., 0.),
    "spindle": (255., 0., 0.),
    "healthy_epithelial": (255., 100., 0.),
    "malignant_epithelial": (100., 100., 0.),
    "fibroblast": (0., 255., 0.),
    "muscle": (0., 255., 255.),
    "endothelial": (200., 30., 20.),
    "neoplastic": (0., 200., 100.),
    "connective": (255., 0., 0.),
    "dead": (255., 0., 255.),
    "nuclei":(255., 255., 255.),
    "squamous_epithel": (220., 220., 55.),
    "glandular_epithel": (0., 110., 155.),
}


NUM_COLORS = {
    0: (255., 0., 55.),
    1: (255., 0., 0.),
    2: (0., 200., 100.),
    3: (220., 220., 55.),
    4: (0., 110., 155.),
    5: (50., 50., 155.),
    6: (220., 220., 55.),
    7: (200., 50., 55.),
    8: (155., 110., 155.),
}


# Adapted from:
# https://github.com/vqdang/hover_net/blob/master/src/misc/viz_utils.py
def draw_thing_contours(
        inst_map: np.ndarray, 
        image: np.ndarray,
        type_map: Optional[np.ndarray]=None,
        fill_contours: Optional[bool]=False,
        thickness: int=2,
        classes: Optional[Dict[str, int]]=None
    ) -> np.ndarray:
    """
    Find coloured contours for a mask and superimpose it on the original
    image mask needs to be instance labelled.

    Args:
    ---------
        inst_map (np.ndarray): 
            Instance segmentation map. Shape (H, W).
        image (np.ndarray): 
            Original image
        type_map (np.ndarray, optional, default=None): 
            Semantic segmentation map. Shape (H, W)
        fill_contours (bool, optional, default=False): 
            If True, contours are filled
        thickness (int, default=2): 
            Thickness of the contour lines
        classes (Dict[str, int], optional, default=None):
            classes dict e.g. {background:0, epithelial:1, ...}
    
    Returns:
    ---------
        np.ndarray: The contours overlaid on top of original image. 
        Shape (H, W).
    """
    bg = np.copy(image)
    
    shape = inst_map.shape[:2]
    nuc_list = list(np.unique(inst_map))

    if 0 in nuc_list:
        nuc_list.remove(0) # 0 is background
    
    for _, nuc_id in enumerate(nuc_list): 
        inst = np.array(inst_map == nuc_id, np.uint8)
        
        y1, y2, x1, x2  = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= shape[0] - 1 else y2
        
        img_crop = image[y1:y2, x1:x2]
        inst_crop = inst[y1:y2, x1:x2]
        inst_bg_crop = bg[y1:y2, x1:x2]
        contours = cv2.findContours(
            inst_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        
        inst_color = None
        if classes is not None:
            class_num = np.unique(type_map[inst > 0].astype("uint8"))[0]
            class_name = [key for key, item in classes.items() if class_num == item][0]

            if class_name in KEY_COLORS.keys():
                inst_color = KEY_COLORS[class_name]
            else:
                inst_color = NUM_COLORS[class_num]

        if fill_contours:
            cv2.drawContours(
                inst_bg_crop, [max(contours, key = cv2.contourArea)], 
                contourIdx=-1, color=inst_color, thickness=-1
            )

            # blend with original image
            alpha = 0.50
            inst_bg_crop = cv2.addWeighted(
                img_crop, 1-alpha, inst_bg_crop, alpha, 0
            )
        else:
            cv2.drawContours(
                inst_bg_crop, contours, contourIdx=-1, 
                color=inst_color, thickness=thickness
            )
            
        bg[y1:y2, x1:x2] = inst_bg_crop

    return bg


def draw_stuff_contours(
        inst_map: np.ndarray, 
        image: np.ndarray,
        type_map: np.ndarray,
        classes: Dict[str, int],
        fill_contours: Optional[bool]=False,
        thickness: int=2,
    ) -> np.ndarray:
    """
    Find coloured contours for a semantic segmentation mask and 
    superimpose it on the original image mask needs to be instance 
    labelled.

    Args:
    ---------
        inst_map (np.ndarray): 
            Instance segmentation map. Shape (H, W).
        image (np.ndarray): 
            Original image
        type_map (np.ndarray): 
            Semantic segmentation map. Shape (H, W)
        classes (Dict[str, int]):
            classes dict e.g. {background:0, epithelial:1, ...}
        fill_contours (bool, optional, default=False): 
            If True, contours are filled
        thickness (int, default=2): 
            Thickness of the contour lines
    
    Returns:
    ---------
        np.ndarray: The contours overlaid on top of original image. 
        Shape (H, W).
    """
    bg = np.copy(image)
    
    obj_list = list(np.unique(inst_map))

    if 0 in obj_list:
        obj_list.remove(0) # 0 is background
    
    for obj in obj_list: 
        inst = np.array(inst_map == obj, np.uint8)
        contours = cv2.findContours(inst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        inst_color = None
        class_num = np.unique(type_map[inst > 0].astype("uint8"))[0]
        class_name = [key for key, item in classes.items() if class_num == item][0]

        if class_name in KEY_COLORS.keys():
            inst_color = KEY_COLORS[class_name]
        else:
            inst_color = NUM_COLORS[class_num]

        if fill_contours:
            cv2.drawContours(
                bg, [max(contours, key=cv2.contourArea)], 
                contourIdx=-1, color=inst_color, thickness=-1
            )
        else:
            cv2.drawContours(
                bg, contours, contourIdx=-1, 
                color=inst_color, thickness=thickness
            )
            
        # blend with original image
        alpha = 0.8
        bg = cv2.addWeighted(image, 1-alpha, bg, alpha, 0)

    return bg
