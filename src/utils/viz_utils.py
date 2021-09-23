import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from matplotlib import pyplot as plt

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
    "nuclei":(255., 0., 0.)
}


# Adapted from https://github.com/vqdang/hover_net/blob/master/src/misc/viz_utils.py
def draw_contours(
        inst_map: np.ndarray, 
        image: np.ndarray,
        type_map: Optional[np.ndarray]=None,
        fill_contours: Optional[bool]=False,
        thickness: int=2,
        classes: Optional[Dict[str, int]]=None
    ) -> np.ndarray:
    """
    Find coloured contours for a mask and superimpose it on the original
    image mask needs to be instance inst_mapled.

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
        classes (Dict[str, int], optional, default=None): c
            lasses dict e.g. {background:0, epithelial:1, ...}
    
    Returns:
    ---------
        np.ndarray: The contours overlaid on top of original image. 
        Shape (H, W).
    """
    bg = np.copy(image)
    
    shape = inst_map.shape[:2]
    nuc_list = list(np.unique(inst_map))
    nuc_list.remove(0) # 0 is background
    
    for idx, nuc_id in enumerate(nuc_list): 
        inst = np.array(inst_map == nuc_id, np.uint8)
        
        y1, y2, x1, x2  = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= shape[0] - 1 else y2
        
        inst_crop = inst[y1:y2, x1:x2]
        inst_bg_crop = bg[y1:y2, x1:x2]
        contours, hierarchy = cv2.findContours(
            inst_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # print(classes)
        inst_color = None
        if classes is not None:
            #clumsy
            class_nums = np.unique(type_map[inst_map > 0].astype("uint8"))
            for key, val in classes.items():
                for num in class_nums:
                    if val == num:
                        inst_color = KEY_COLORS[key]
            

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

    return bg


def viz_patches(patches: np.ndarray) -> Tuple[int]:
    """
    patches is assumed to be of shape (n_patches, H, W, n_channels)
    This function vizualizes those patches. Don't put too many patches
    in or everything willl crash.

    Args:
    ----------
        patches (np.ndarray): 
            numpy array of stacked image patches. Shape: 
            (n_patches, H, W, C)

    Returns:
    ----------
        Tuple: Shape of the patches array
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


def viz_instance(inst_map: np.ndarray, ix: int = 1) -> Tuple[int]:
    """
    This function will visualize a single instance with id 'ix' from the
    'inst_map'

    Args:
    ----------
        inst_map (np.ndarray): 
            the instance map
        ix (int): 
            the index/id of an instance

    Returns:
    -----------
        Shape of the patches array
    """

    nuc_map = np.copy(inst_map == ix)
    y1, y2, x1, x2 = bounding_box(nuc_map)
    y1 = y1 - 2 if y1 - 2 >= 0 else y1
    x1 = x1 - 2 if x1 - 2 >= 0 else x1
    x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
    y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
    nuc_map_crop = nuc_map[y1:y2, x1:x2].astype("int32")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(nuc_map_crop)

    return nuc_map_crop.shape
