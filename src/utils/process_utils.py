"""
Utility functions mainly ported from HoVer-Net repo
"""

import cv2
import numpy as np
import scipy.ndimage as ndi

from typing import List, Tuple


def center_crop(img: np.ndarray, ch: int, cw: int) -> np.ndarray:
    """
    Center crop an input image

    Args:
        img (np.ndarray): 
            Input img. Shape (H, W).
        ch (int):
            Crop height
        cw (int):
            crop width
    """
    if len(img.shape) == 3:
        H, W, _ = img.shape
    else:
        H, W = img.shape

    x = W // 2 - (cw // 2)
    y = H // 2 - (ch // 2)    
    img = img[y:y + ch, x:x + cw, :] if len(img.shape) == 3 else img[y:y + ch, x:x + cw]
    return img
    

# Ported from https://github.com/vqdang/hover_net/blob/master/src/misc/utils.py
def bounding_box(inst_map: np.ndarray) -> List[int]:
    """
    Bounding box coordinates for nuclei instance
    that is given as input. This assumes that the inst_map 
    has only one instance in it.

    Args:
        inst_map (np.ndarray): instance labels
    """
    rows = np.any(inst_map, axis=1)
    cols = np.any(inst_map, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


# ported from https://github.com/vqdang/hover_net/tree/master/src/metrics/sample
def remap_label(pred: np.ndarray, by_size: bool = False) -> np.ndarray:
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred (np.ndarray):
            the 2d array contain instances where each instances is marked
            by non-zero integer
        by_size (bool, default=False): 
            renaming with larger nuclei has smaller id (on-top)
    
    Returns:
        np.ndarray inst map with remapped contiguous labels
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1    
    return new_pred


# Ported from https://github.com/vqdang/hover_net/blob/master/src/misc/utils.py
def get_inst_centroid(inst_map: np.ndarray) -> np.ndarray:
    """
    Get centroid x, y coordinates from each unique nuclei instance

    Args:
        inst_map (np.ndarray): nuclei instance map

    Returns:
        an np.ndarray of shape (num_instances, 2)

        Example:
            array([[780.05089286, 609.11741071],
                   [890.64603817, 237.89589358],
                   [944.37971014, 541.3942029 ],
                   ...,
                   [ 77.5       , 536.        ],
                   [ 78.21428571, 541.64285714],
                   [485.        , 893.        ]])
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                         (inst_moment["m01"] / inst_moment["m00"])]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


def get_inst_types(inst_map: np.ndarray, type_map: np.ndarray) -> np.ndarray:
    """
    Get the types of every single instance in an instance map
    and write them to a 1D-Array
    
    Args:
        inst_map (np.ndarray): instance map of shape (H, W)
        type_map (np.ndarray): type map of shape (H, W). Labels are indices.

    Returns:
        an np.ndarray of shape (num_instances, 1)

        Example:
            array([[3],
                   [3],
                   [3],
                   ...,
                   [1],
                   [1],
                   [1]], dtype=int32)
    """
    inst_ids = list(np.unique(inst_map))
    inst_ids.remove(0)
    inst_types = np.full((len(inst_ids), 1), 0, dtype=np.int32)
    for j, id_ in enumerate(inst_ids):
        inst_type = np.unique(type_map[inst_map == id_])[0]
        inst_types[j] = inst_type
    return inst_types


def instance_contours(inst_map: np.ndarray, thickness: int = 2):
    """
    Find a contour for each nuclei instance in a mask

    Args:
        inst_map (np.ndarray): instance map
        thickness (int): thickness of the contour line
    """

    # Padding first to avoid contouring the image borders
    inst_map2 = np.pad(inst_map.copy(), ((thickness, thickness), (thickness, thickness)), 'edge')
    bg = np.zeros(inst_map2.shape, dtype=np.uint8)
    for j, nuc_id in enumerate(np.unique(inst_map)):
        inst_map = np.array(inst_map2 == nuc_id, np.uint8)
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_bg_crop = bg[y1:y2, x1:x2]
        contours, hierarchy = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contoured_rgb = cv2.drawContours(
            inst_bg_crop, contours, contourIdx=-1,
            color=(255, 255, 255), thickness=thickness
        )

        bg[y1:y2, x1:x2] = inst_bg_crop

    bg = bg[thickness:-thickness, thickness:-thickness]
    return bg, contours


def get_type_instances(inst_map: np.ndarray,
                       type_map: np.ndarray,
                       class_num: int) -> np.ndarray:
    """
    Get the instances from an instance map that belong to class 'class_num'
    Drop everything else. The type map and inst map need to have the exact same 
    non-zero pixels.
    
    Args:
        inst_map (np.ndarray): instance map of shape (H, W)
        type_map (np.ndarray): type map of shape (H, W). Labels are indices.
        class_num (int): class label  
    """
    t = type_map.astype("uint8") == class_num
    imap = np.copy(inst_map)
    imap[~t] = 0
    return imap


def one_hot(type_map: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert type map of shape (H, W) to one hot encoded types of shape (H, W, C)
    
    Args:
        type_map (np.ndarray): type map of shape (H, W). Labels are indices.
        num_classes (int): number of classes in the dataset
    """
    return np.eye(num_classes+1)[type_map]


def overlays(im: np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
    Overlay mask with original image where there is no background
    mask is assumed to have shape (HxW)

    Args:
        im (np.ndarray): original image of shape (H, W, C)
        mask (np.ndarray): instance or type mask of the nucleis in the image
    """
    return np.where(mask[..., None], im, 0)


def type_map_flatten(type_map: np.ndarray) -> np.ndarray:
    """
    Convert a one hot type map of shape (H, W, C) to a single channel
    indice map of shape (H, W)
    
    Args:
        type_map (np.ndarray): type_map to be flattened
    """
    type_out = np.zeros([type_map.shape[0], type_map.shape[1]])
    for t in np.unique(type_map):
        type_tmp = type_map == t
        type_out += (type_tmp * t)
    return type_out