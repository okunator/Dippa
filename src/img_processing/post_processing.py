import torch
import scipy
import cv2
import numpy as np
import pandas as pd
import skimage.morphology as morph
import skimage.segmentation as segm
from skimage.exposure import histogram
from skimage.filters import difference_of_gaussians
from skimage import filters
from scipy.spatial import distance_matrix
from torch import nn
from scipy import ndimage as ndi

from src.img_processing.process_utils import (
    bounding_box, get_inst_centroid, binarize
)


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
    Also Effectively removes checkerboard effect after predicted tiles are merged
    and makes the thresholding of the soft maps trivial with the histogram. However
    the metrics might get worse...
    Args:
        prob_map (np.ndarray): logit or probability map of shape (H, W) 
    """
    prob_map = difference_of_gaussians(prob_map, 1, 50)
    prob_map = activation(prob_map, 'relu')
    prob_map = difference_of_gaussians(prob_map, 1, 10)
    prob_map = activation(prob_map, 'sigmoid')
    return prob_map


def to_inst_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Takes in a binary mask -> fill holes -> removes small objects -> label connected components
    Args:
        binary_mask (np.ndarray): a binary mask to be labelled. Shape (H, W)
    """
    mask = ndi.binary_fill_holes(binary_mask)
    mask = morph.remove_small_objects(binary_mask.astype(bool), min_size=64)
    inst_map = ndi.label(mask)[0]
    return inst_map


def naive_thresh_prob(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Threshold an activated soft mask with probability values [0, 1].
    Args:
        prob_map (np.ndarray): soft mask to be thresholded. Shape (H, W)
        threshold (float): thresholding cutoff between [0, 1]
    """
    assert 0 <= threshold <= 1, f"thresh = {threshold}. given threshold not between [0,1]"
    seg = prob_map.copy()
    seg = seg > threshold
    inst_map = to_inst_map(seg)
    return inst_map


def naive_thresh(prob_map: np.ndarray, threshold: int = 2) -> np.ndarray:
    """
    Threshold a soft mask. Values can be logits or probabilites
    Args:
        prob_map (np.ndarray): soft mask to be thresholded. Shape (H, W)
        threshold (int): value used to divide the max value of the mask
    """
    seg = prob_map.copy()
    seg[seg < np.amax(prob_map)/threshold] = 0
    seg[seg > np.amax(prob_map)/threshold] = 1
    inst_map = to_inst_map(seg)
    return inst_map


def niblack_thresh(prob_map: np.ndarray, win_size: int = 13) -> np.ndarray:
    """
    Wrapper for skimage niblack thresholding method
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html
    
    Args:
        prob_map (np.ndarray): soft mask to be thresholded. Shape (H, W)
        win_size (int): size of the window used in the thresholding algorithm
    """
    thresh = filters.threshold_niblack(prob_map, window_size=win_size)
    mask = prob_map > thresh
    mask = cv2_opening(mask)
    inst_map = to_inst_map(mask)
    return inst_map
    

def sauvola_thresh(prob_map, win_size=33):
    """
    Wrapper for skimage sauvola thresholding method
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html

    Args:
        prob_map (np.ndarray): soft mask to be thresholded. Shape (H, W)
        win_size (int): size of the window used in the thresholding algorithm
    """
    thresh = filters.threshold_sauvola(prob_map, window_size=win_size)
    mask = prob_map > thresh
    mask = cv2_opening(mask)
    inst_map = to_inst_map(mask)
    return inst_map


def morph_chan_vese_thresh(prob_map: np.ndarray) -> np.ndarray:
    """
    Morphological chan vese method for thresholding. Skimage wrapper
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese
    Args: 
        prob_map (np.ndarray): soft mask to be thresholded. Shape (H, W)
    """
    init_ls = segm.checkerboard_level_set(prob_map.shape, 2)
    ls = segm.morphological_chan_vese(prob_map, 35, smoothing=1, init_level_set=init_ls)
    hist = np.histogram(ls)[0]
    
    if hist[-1] > hist[0]:
        ls = 1 - ls
        
    inst_map = to_inst_map(ls)
    return inst_map


def smoothed_thresh(prob_map: np.ndarray) -> np.ndarray:
    """
    Thresholding probability map after it has been smoothed with gaussian differences
    After dog the prob_map histogram has a notable discontinuity which can be found by
    taking the minimum of the derivative of the histogram -> no need for arbitrary cutoff
    value for threshold.

    Args:
        prob_map (np.ndarray): soft mask to be thresholded. Shape (H, W)
    """
    # Find the steepest drop in the histogram
    hist, hist_centers = histogram(prob_map)
    d = np.diff(hist)
    b = d == np.min(d)
    b = np.append(b, False) # append one since np.diff loses one element in arr
    thresh = hist_centers[b][0] + 0.07
    mask = naive_thresh_prob(prob_map, thresh)
    return mask


def cv2_opening(mask: np.ndarray) -> np.ndarray:
    """
    Takes in an inst_map -> binarize -> apply morphological opening (2 iterations) -> label
    Seems to increase segmentation metrics
    Args:
        mask (np.ndarray): instance map to be opened. Shape (H, W)
    """
    mask = binarize(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_mask = (mask*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    inst_map = ndi.label(new_mask)[0]
    return inst_map



def cv2_closing(mask: np.ndarray) -> np.ndarray:
    """
    Takes in an inst_map -> binarize -> apply morphological closing (2 iterations) -> label
    Args:
        mask (np.ndarray): instance map to be opened. Shape (H, W)
    """
    mask = binarize(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_mask = (mask*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    inst_map = ndi.label(new_mask)[0]
    return inst_map


def sobel_watershed(prob_map: np.ndarray,
                    inst_map: np.ndarray,
                    win_size: int = 13) -> np.ndarray:
    """
    After thresholding, this function can be used to compute distance maps for each nuclei instance
    and watershed segment the elevation map of the prob_map (aobel). Before computing distance maps
    a binary opening is performed to the instance map (seems to increase metrics). Markers for the
    distance maps are computed using niblack thresholding on the distance map.

    Args:

        inst_map (np.ndarray): The instance map to be segmented. Shape (H, W)
        win_size (int): window size used in niblack thresholding the distance maps to
                        find markers for watershed
    """
    # Morphological opening for smoothing the likely jagged edges
    # This typically improves PQ
    seg = np.copy(inst_map)
    new_mask = cv2_opening(seg)
    ann = ndi.label(new_mask)[0]

    shape = seg.shape[:2]
    nuc_list = list(np.unique(ann))
    nuc_list.remove(0)

    # find the distance map per nuclei instance
    distmap = np.zeros(shape, dtype=np.uint8)
    for nuc_id in nuc_list:
        nuc_map = np.copy(ann == nuc_id)
        
        # Do operations to the bounded box of the nuclei
        # rather than the full size matrix
        y1, y2, x1, x2  = bounding_box(nuc_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= ann.shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= ann.shape[0] - 1 else y2
        nuc_map_crop = nuc_map[y1:y2, x1:x2]

        # Find distance transform of the bounding box and normalize it
        distance = ndi.distance_transform_edt(nuc_map_crop)
        distance = 255 * (distance / np.amax(distance))
        distmap[y1:y2, x1:x2] += distance.astype('uint8')
                
    # find markers for ws
    markers = np.copy(distmap)
    markers = niblack_thresh(distmap, win_size)
    
    elevation_map = filters.sobel(prob_map)
    inst_map = segm.watershed(elevation_map, markers, mask=ann, watershed_line=True)
    
    # remove small objs. Will enhance PQ a lot '(HACKish)'
    # mask[mask > 0] = 1
    # mask = morph.remove_small_objects(mask.astype(bool), min_size=100)
    # inst_map = ndi.label(mask)[0]
    
    return inst_map


# adapted from https://github.com/vqdang/hover_net/blob/master/src/postproc/other.py
def inv_dist_watershed(inst_map: np.ndarray, win_size: int = 13) -> np.ndarray:
    """
    After thresholding, this function can be used to compute distance maps for each nuclei instance
    and watershed segment the inverse distmaps. Before computing distance maps a binary opening
    is performed to the instance map (seems to increase metrics). Markers for the distance maps
    are computed using niblack thresholding on the distance map.

    Args:
        inst_map (np.ndarray): The instance map to be segmented
        win_size (int): window size used in niblack thresholding the 
                        distance maps to find markers for watershed
    """
    
    seg = np.copy(inst_map)
    new_mask = cv2_opening(seg)
    ann = ndi.label(new_mask)[0]

    shape = seg.shape[:2]
    nuc_list = list(np.unique(ann))
    nuc_list.remove(0)

    distmap = np.zeros(shape, dtype=np.uint8)
    for nuc_id in nuc_list:
        nuc_map = np.copy(ann == nuc_id)
        
        # Do operations to the bounded box of the nuclei
        # rather than the full size matrix
        y1, y2, x1, x2  = bounding_box(nuc_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= ann.shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= ann.shape[0] - 1 else y2
        nuc_map_crop = nuc_map[y1:y2, x1:x2]

        distance = ndi.distance_transform_edt(nuc_map_crop)
        distance = 255 * (distance / np.amax(distance))
        distmap[y1:y2, x1:x2] += distance.astype('uint8')
                
    markers = np.copy(distmap)
    markers = niblack_thresh(distmap, win_size)
                    
    # watershed
    inst_map = segm.watershed(-distmap, markers, mask=ann, watershed_line=True)
    
    # remove small cells. Will enhance PQ in kumar a lot (HACKish)
    # mask[mask > 0] = 1
    # mask = morph.remove_small_objects(mask.astype(bool), min_size=100)
    # inst_map = ndi.label(mask)[0]
    
    return inst_map


# Adapted from https://github.com/vqdang/hover_net/blob/master/src/process.py
def combine_inst_semantic(inst_map: np.ndarray,
                          type_map: np.ndarray) -> np.ndarray:
    """
    Takes in the outputs of the different segmentation heads and combines them into
    one panoptic segmentation result
    Args:
        inst_map (np.ndarray): output from the instance segmentation head of 
                               a panoptic model or the post processed output of 
                               the instance seg head. Shape (H, W)
        type_map (np.ndarray): output from the type segmentation head of 
                               a panoptic model Shape (H, W)
    """
    inst_ids = {}
    pred_id_list = list(np.unique(inst_map))[1:]  # exclude background ID
    for inst_id in pred_id_list:
        inst_tmp = inst_map == inst_id
        inst_type = type_map[inst_map == inst_id].astype("uint16")
        inst_centroid = get_inst_centroid(inst_tmp)
        bbox = bounding_box(inst_tmp)
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
            else:
                inst_type = "bg"

        inst_ids[inst_id] = {
            "bbox": bbox,
            "centroid_x": inst_centroid[0][0],
            "centroid_y": inst_centroid[0][1],
            "inst_type": inst_type,
            "type_list": type_list
        }


    # If inst_type == "bg" (instance has no label) find the nearest instance with
    # a class label and use that. This way no instances are dropped from the end result
    # pandas wrangling got pretty fugly..
    inst_df = pd.DataFrame(inst_ids).transpose()
    cm = inst_df[["centroid_x", "centroid_y", "inst_type"]]
    found_centroids = cm[cm["inst_type"] != "bg"]
    missed_centroids = cm[cm["inst_type"] == "bg"]
    mc = missed_centroids[["centroid_x", "centroid_y"]].values
    fc = found_centroids[["centroid_x", "centroid_y"]].values
    centroid_dists = distance_matrix(mc, fc)
    closest_centroids = fc[np.argmin(centroid_dists, axis=1)]

    cc_df = pd.DataFrame(closest_centroids, columns=["centroid_x", "centroid_y"])
    cc_df = found_centroids.merge(cc_df, on=["centroid_x", "centroid_y"], how="inner")
    missed_centroids = missed_centroids.assign(inst_type=cc_df["inst_type"].values)
    inst_df = inst_df.merge(missed_centroids, on=["centroid_x", "centroid_y"], how="left")
    inst_df.index += 1
    inst_ids = inst_df.transpose().to_dict()

    type_map_out = np.zeros([type_map.shape[0], type_map.shape[1]])
    for inst_id, val in inst_ids.items():
        inst_tmp = inst_map == inst_id
        inst_type = type_map[inst_map == inst_id].astype("uint16")
        inst_type = val["inst_type_y"] if val["inst_type_x"] == "bg" else val["inst_type_x"]
        type_map_out += (inst_tmp * int(inst_type))

    return type_map_out
