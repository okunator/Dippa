import numpy as np
import math
import random
import colorsys
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt
from src.img_processing.process_utils import *

# ported from https://github.com/vqdang/hover_net/blob/master/src/misc/viz_utils.py
def random_colors(N: int, bright: bool = True) -> List[Tuple[float]]:
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    Args:
        N (int): number of unique instances in the inst map
        bright (bool): If True brightness is higher
    Returns:
        List of rgb color tuples
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)

    return colors

# ported from https://github.com/vqdang/hover_net/blob/master/src/misc/viz_utils.py
# minor mods
def draw_contours(mask: np.ndarray, 
                  image: np.ndarray,
                  type_map: np.ndarray = None,
                  fill_contours: bool = False,
                  thickness: int = 2) -> Tuple[np.ndarray]:
    """
    Find contours for rgb mask to superimpose it the original image
    mask needs to be instance labelled.
    Args: 
        mask (np.ndarray): inst_map
        image (np.ndarray): image
        type_map (np.ndarray): type_map
        fill_contours (bool): If True, contours are filled
        thickness (int): thickness ofthe contour line
    Returns:
        The background? and contours overlaid on top of original image.
    """
    bg = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    
    shape = mask.shape[:2]
    nuc_list = list(np.unique(mask))
    nuc_list.remove(0) # 0 is background
    
    if type_map is None:
        inst_colors = random_colors(len(nuc_list))
        inst_colors = np.array(inst_colors)
    else:
        inst_colors = np.array([
            (255., 0., 0.), (0., 255., 0.), (0., 0., 255.),
            (255., 255., 0.), (255., 165., 0.), (0., 255., 255.)
        ])
    
    for idx, nuc_id in enumerate(nuc_list): 
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

        k = idx if type_map is None else np.unique(type_map[inst_map > 0])[0]
        inst_color = inst_colors[k]
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


def viz_patches(patches: np.ndarray) -> Tuple[int]:
    """
    patches is assumed to be of shape (n_patches, H, W, n_channels)
    """
    fignum = 200
    low=0
    high=len(patches)

    # Visualize
    fig_patches = plt.figure(fignum, figsize=(35,35))
    pmin, pmax = patches.min(), patches.max()
    dims = np.ceil(np.sqrt(high - low))
    for idx in range(high - low):
        spl = plt.subplot(dims, dims, idx + 1)
        ax = plt.axis("off")
        imm = plt.imshow(patches[idx].astype("uint8"))
        cl = plt.clim(pmin, pmax)
    plt.show()
    return patches.shape
