import math
import random
import colorsys
from .process_utils import *

# ported from https://github.com/vqdang/hover_net/blob/master/src/misc/viz_utils.py
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)

    return colors

# ported from https://github.com/vqdang/hover_net/blob/master/src/misc/viz_utils.py
# minor mods
def draw_contours(mask, image, fill_contours=False, thickness=2):
    # Find contours for rgb mask to superimpose it the original image
    # mask needs to be instance labelled
    bg = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    
    shape = mask.shape[:2]
    nuc_list = list(np.unique(mask))
    nuc_list.remove(0) # 0 is background
    inst_colors = random_colors(len(nuc_list))
    inst_colors = np.array(inst_colors)
    
    for idx, nuc_id in enumerate(nuc_list):
        inst_color = inst_colors[idx]
        inst_map = np.array(mask == nuc_id, np.uint8)
        y1, y2, x1, x2  = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_bg_crop = bg[y1:y2, x1:x2]
        contours, hierarchy = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if fill_contours:
            contoured_rgb = cv2.drawContours(
                inst_bg_crop, [max(contours, key = cv2.contourArea)], 
                contourIdx=-1, color=inst_color, thickness=-1
            )
        else:
            contoured_rgb = cv2.drawContours(
                inst_bg_crop, contours, contourIdx=-1, 
                color=inst_color, thickness=thickness
            )
            
        bg[y1:y2, x1:x2] = inst_bg_crop
    
    image_overlayed = np.where(bg, image, 255)

    return bg, image_overlayed
