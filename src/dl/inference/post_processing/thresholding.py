import numpy as np
import skimage.segmentation as segm
import skimage.filters as filters
from skimage.exposure import histogram

from src.utils.mask_utils import to_inst_map, cv2_opening


def naive_thresh_prob(prob_map: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
    """
    Threshold a sigmoid/softmax activated soft mask.

    Args:
    ----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        threshold (float, default=0.5): 
            Thresholding cutoff between [0, 1]
    """
    assert 0 <= threshold <= 1, f"thresh = {threshold}. given threshold not between [0,1]"
    seg = prob_map.copy()
    seg = seg >= threshold
    inst_map = to_inst_map(seg)
    return inst_map


def naive_thresh(prob_map: np.ndarray, threshold: int = 2, **kwargs) -> np.ndarray:
    """
    Threshold a soft mask. Values can be logits or probabilites

    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        threshold (int, default=2): 
            Value used to divide the max value of the mask
    """
    seg = prob_map.copy()
    seg[seg < np.amax(prob_map)/threshold] = 0
    seg[seg > np.amax(prob_map)/threshold] = 1
    inst_map = to_inst_map(seg)
    return inst_map


def niblack_thresh(prob_map: np.ndarray, win_size: int = 13, **kwargs) -> np.ndarray:
    """
    Wrapper for skimage niblack thresholding method
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html
    
    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        win_size (int, default=13): 
            Size of the window used in the thresholding algorithm
    """
    thresh = filters.threshold_niblack(prob_map, window_size=win_size)
    mask = prob_map > thresh
    mask = cv2_opening(mask)
    inst_map = to_inst_map(mask)
    return inst_map
    

def sauvola_thresh(prob_map: np.ndarray, win_size: int=33, **kwargs) -> np.ndarray:
    """
    Wrapper for skimage sauvola thresholding method
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html

    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        win_size (int, default=33):
            Size of the window used in the thresholding algorithm
    """
    thresh = filters.threshold_sauvola(prob_map, window_size=win_size)
    mask = prob_map > thresh
    mask = cv2_opening(mask)
    inst_map = to_inst_map(mask)
    return inst_map


def morph_chan_vese_thresh(prob_map: np.ndarray, **kwargs) -> np.ndarray:
    """
    Morphological chan vese method for thresholding. Skimage wrapper
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese

    Args: 
    -----------
        prob_map (np.ndarray): 
            soft mask to be thresholded. Shape (H, W)
    """
    init_ls = segm.checkerboard_level_set(prob_map.shape, 2)
    ls = segm.morphological_chan_vese(prob_map, 35, smoothing=1, init_level_set=init_ls)
    hist = np.histogram(ls)[0]
    
    if hist[-1] > hist[0]:
        ls = 1 - ls
        
    inst_map = to_inst_map(ls)
    return inst_map


def argmax(prob_map: np.ndarray, **kwargs) -> np.ndarray:
    """
    Wrapper to take argmax of a one_hot logits or prob map

    Args:
    -----------
        prob_map (np.ndarray): 
            The probability map of shape (H, W, C)

    Returns:
        a mask of indices shaped (H, W)
    """
    return to_inst_map(np.argmax(prob_map, axis=2))


def smoothed_thresh(prob_map: np.ndarray, eps: float = 0.01, **kwargs) -> np.ndarray:
    """
    Thresholding probability map after it has been smoothed with gaussian differences
    After dog the prob_map histogram has a notable discontinuity which can be found by
    taking the minimum of the derivative of the histogram -> no need for arbitrary cutoff
    value for threshold.

    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        eps (int): 
            Small increase the threshold, since the hist center
            often not optimal
    """
    # Find the steepest drop in the histogram
    hist, hist_centers = histogram(prob_map)
    d = np.diff(hist)
    b = d == np.min(d)
    b = np.append(b, False) # append one since np.diff loses one element in arr
    thresh = hist_centers[b][0] + eps
    mask = naive_thresh_prob(prob_map, thresh)
    return mask
