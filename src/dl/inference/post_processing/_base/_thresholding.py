import numpy as np
import skimage.segmentation as segm
import skimage.filters as filters
from skimage.exposure import histogram

from src.utils import polish_mask, cv2_opening


__all__ = [
    "naive_thresh_prob", "naive_thresh", "niblack_thresh", "sauvola_thresh",
    "morph_chan_vese_thresh", "smoothed_thresh", "argmax"
]


def naive_thresh_prob(
        prob_map: np.ndarray,
        threshold: float=0.5,
        **kwargs
    ) -> np.ndarray:
    """
    Threshold a sigmoid/softmax activated soft mask.

    Args:
    ----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        threshold (float, default=0.5): 
            Thresholding cutoff between [0, 1]

    Returns:
    -----------
        np.ndarray: Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    assert 0 <= threshold <= 1, (
        f"thresh = {threshold}. given threshold not between [0,1]"
    )
    seg = prob_map.copy()
    seg = seg >= threshold
    seg = polish_mask(seg)

    return seg


def naive_thresh(
        prob_map: np.ndarray, 
        threshold: int=2, 
        **kwargs
    ) -> np.ndarray:
    """
    Threshold a soft mask. Values can be logits or probabilites

    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        threshold (int, default=2): 
            Value used to divide the max value of the mask

    Returns:
    -----------
        np.ndarray: Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    seg = prob_map.copy()
    seg[seg < np.amax(prob_map)/threshold] = 0
    seg[seg > np.amax(prob_map)/threshold] = 1
    seg = polish_mask(seg)

    return seg

def niblack_thresh(
        prob_map: np.ndarray,
        win_size: int=13,
        **kwargs
    ) -> np.ndarray:
    """
    Wrapper for skimage niblack thresholding method
    
    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        win_size (int, default=13): 
            Size of the window used in the thresholding algorithm

    Returns:
    -----------
        np.ndarray: Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    thresh = filters.threshold_niblack(prob_map, window_size=win_size)
    mask = prob_map > thresh
    seg = cv2_opening(mask)
    seg = polish_mask(seg)

    return seg
    

def sauvola_thresh(
        prob_map: np.ndarray,
        win_size: int=33,
        **kwargs
    ) -> np.ndarray:
    """
    Wrapper for skimage sauvola thresholding method

    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        win_size (int, default=33):
            Size of the window used in the thresholding algorithm

    Returns:
    -----------
        np.ndarray: Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    thresh = filters.threshold_sauvola(prob_map, window_size=win_size)
    mask = prob_map > thresh
    seg = cv2_opening(mask)
    seg = polish_mask(seg)

    return seg


def morph_chan_vese_thresh(prob_map: np.ndarray, **kwargs) -> np.ndarray:
    """
    Morphological chan vese method for thresholding. Skimage wrapper

    Args: 
    -----------
        prob_map (np.ndarray): 
            soft mask to be thresholded. Shape (H, W)

    Returns:
    -----------
        np.ndarray: Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    init_ls = segm.checkerboard_level_set(prob_map.shape, 2)
    ls = segm.morphological_chan_vese(
        prob_map, 35, smoothing=1, init_level_set=init_ls
    )

    hist = np.histogram(ls)[0]
    if hist[-1] > hist[0]:
        ls = 1 - ls
        
    seg = polish_mask(ls)

    return seg


def argmax(prob_map: np.ndarray, **kwargs) -> np.ndarray:
    """
    Wrapper to take argmax of a one_hot logits or prob map

    Args:
    -----------
        prob_map (np.ndarray): 
            The probability map of shape (H, W, C)|(H, W)

    Returns:
    -----------
        np.ndarray: a mask of argmax indices. Shape: (H, W). Type: uint8.
    """
    if len(prob_map.shape) == 2:
        inv_prob = 1 - prob_map
        prob_map = np.stack([inv_prob, prob_map], axis=-1)
        
    seg = np.argmax(prob_map, axis=-1).astype("u4")

    return seg


def smoothed_thresh(
        prob_map: np.ndarray,
        eps: float=0.01,
        **kwargs
    ) -> np.ndarray:
    """
    Thresholding probability map after it has been smoothed with 
    gaussian differences. After dog, the prob_map histogram has a 
    notable discontinuity which can be found by taking the minimum of 
    the derivative of the histogram -> no need for arbitrary cutoff
    value for threshold.

    Args:
    -----------
        prob_map (np.ndarray): 
            Soft mask to be thresholded. Shape (H, W)
        eps (float, default=0.01): 
            Small increase the threshold, since the hist center
            often not optimal

    Returns:
    -----------
        np.ndarray: Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    # Find the steepest drop in the histogram
    hist, hist_centers = histogram(prob_map)
    d = np.diff(hist)
    b = d == np.min(d)
    b = np.append(b, False) # append 1 b/c np.diff loses one elem in arr
    thresh = hist_centers[b][0] + eps
    seg = naive_thresh_prob(prob_map, thresh)
    
    return seg
