import cv2
import torch
import torch.nn as nn
import numpy as np
import skimage.morphology as morph
import skimage.filters as filters
import scipy.ndimage as ndi


def binarize(inst_map: np.ndarray) -> np.ndarray:
    """
    Binarize a labelled instance map

    Args:
        inst_map (np.ndarray): instance map to be binarized
    """
    inst_map[inst_map > 0] = 1
    return inst_map.astype("uint8")


def activation(prob_map: np.ndarray, method: str = 'sigmoid') -> np.ndarray:
    """
    torch activations functions for logit/soft mask that has been converted to 
    np.ndarray array

    Args:
        prob_map (np.ndarray): the soft mask to be activated. shape (H, W)
        method (str): one of ('relu', 'celu', 'sigmoid', 'relu-sigmoid', 'celu-sigmoid', 'None')  
    """
    assert method in ('relu', 'celu', 'sigmoid', 'relu-sigmoid', 'celu-sigmoid', 'None')
    if method == 'relu':
        act = torch.from_numpy(prob_map).relu().numpy()
    elif method == 'celu':
        celu = nn.CELU()
        act = celu(torch.from_numpy(prob_map)).numpy()
    elif method == 'sigmoid':
        act = torch.from_numpy(prob_map).sigmoid().numpy()
    elif method == 'relu-sigmoid':
        act = torch.from_numpy(prob_map).relu().sigmoid().numpy()
    elif method == 'celu-sigmoid':
        celu = nn.CELU()
        act = celu(torch.from_numpy(prob_map)).sigmoid().numpy()
    else:
        act = prob_map

    return act


def activate_plus_dog(prob_map: np.ndarray) -> np.ndarray:
    """
    Takes in a logit or prob map and applies DoG and activation twice to it.
    This can remove artifacts from the prob map but some information will be lost
    Removes checkerboard effect after predicted tiles are merged and makes 
    the thresholding of the soft maps trivial with the histogram.

    Args:
        prob_map (np.ndarray): logit or probability map of shape (H, W) 
    """
    prob_map = filters.difference_of_gaussians(prob_map, 1, 50)
    prob_map = activation(prob_map, 'relu')
    prob_map = filters.difference_of_gaussians(prob_map, 1, 10)
    prob_map = activation(prob_map, 'sigmoid')
    return prob_map


def to_inst_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Takes in a binary mask -> fill holes -> removes small objects -> label connected components
    If class channel is included this assumes that binary_mask[..., 0] is the bg channel and
    binary_mask[..., 1] the foreground.

    Args:
        binary_mask (np.ndarray): a binary mask to be labelled. Shape (H, W) or (H, W, C)
    
    Returns:
        labelled instances np.ndarray of shape (H, W)
    """
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[..., 1]

    mask = ndi.binary_fill_holes(binary_mask)
    mask = morph.remove_small_objects(binary_mask.astype(bool), min_size=10)
    inst_map = ndi.label(mask)[0]

    return inst_map


def cv2_opening(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Takes in an inst_map -> binarize -> apply morphological opening (2 iterations) -> label
    Seems to increase segmentation metrics

    Args:
        mask (np.ndarray): instance map to be opened. Shape (H, W)
        iterations: int: number of iterations for the operation
    """
    mask = binarize(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_mask = (mask*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    inst_map = ndi.label(new_mask)[0]
    return inst_map


def cv2_closing(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Takes in an inst_map -> binarize -> apply morphological closing (2 iterations) -> label
    
    Args:
        mask (np.ndarray): instance map to be opened. Shape (H, W)
        iterations: int: number of iterations for the operation
    """
    mask = binarize(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_mask = (mask*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    inst_map = ndi.label(new_mask)[0]
    return inst_map


def remove_debris(inst_map: np.ndarray, min_size: int = 10):
    """
    Remove small objects from an inst map

    Args:
        inst_map (np.ndarray): instance map to be binarized
        min_size (int): min_size for the objects that are left untouched
    """
    res = np.zeros(inst_map.shape, np.int32)
    for ix in np.unique(inst_map)[1:]:
        nuc_map = np.copy(inst_map == ix)
        y1, y2, x1, x2 = bounding_box(nuc_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        nuc_map_crop = nuc_map[y1:y2, x1:x2].astype("int32")
        nuc_map_crop = morph.remove_small_objects(
            nuc_map_crop.astype(bool), min_size, connectivity=1).astype("int32")
        nuc_map_crop[nuc_map_crop > 0] = ix
        res[y1:y2, x1:x2] += nuc_map_crop
    return res