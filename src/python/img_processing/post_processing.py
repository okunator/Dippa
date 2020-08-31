import torch
import scipy
import cv2
import numpy as np
import skimage.morphology as morph
import skimage.segmentation as segm
from skimage import filters
from torch import nn
from scipy import ndimage as ndi
from .process_utils import *


def medfilter_instances(seg):
    # Median filter each unique instance to smoothen the borders of nucleis
    shape = seg.shape[:2]
    ann = ndi.label(seg)[0]
    nuc_list = list(np.unique(ann))
    nuc_list.remove(0) # 0 is background

    filtmap = np.zeros(shape, dtype=np.uint32)    
    for nuc_id in nuc_list:
        nuc_map = np.copy(ann == nuc_id)
        filt = filters.median(nuc_map, disk(2))
        filt[filt > 0] = 1
        filtmap += filt
    
    return filtmap


def activation(prob_map, method='sigmoid'):
    # Activation
    assert method in ('relu', 'sigmoid', 'relu-sigmoid', 'celu-sigmoid', 'None')
    if method == 'relu':
        act = torch.from_numpy(prob_map).relu().numpy()
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


def naive_thresh_logits(prob_map, threshold=0.5):
    # Threshold naively when values b/w [0,1]
    seg = prob_map.copy()
    seg = seg > threshold
    
    # Final smoothening of the jagged edges
    mask = scipy.signal.medfilt(seg, 3)
    mask = ndi.binary_fill_holes(mask)
    mask = morph.remove_small_objects(seg.astype(bool), min_size=64)
    inst_map = ndi.label(mask)[0]
    return inst_map


def naive_thresh(prob_map, threshold=2):
    # Threshold naively 
    seg = prob_map.copy()
    seg[seg < np.amax(prob_map)/threshold] = 0
    seg[seg > np.amax(prob_map)/threshold] = 1
    
    # Final smoothening of the jagged edges
    mask = scipy.signal.medfilt(seg, 3)
    mask = ndi.binary_fill_holes(mask)
    mask = morph.remove_small_objects(seg.astype(bool), min_size=64)
    inst_map = ndi.label(mask)[0]
    return inst_map


def niblack_thresh(prob_map, win_size=13):
    thresh = filters.threshold_niblack(prob_map, window_size=win_size)
    mask = prob_map > thresh
    mask = cv2_opening(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = morph.remove_small_objects(mask, min_size=64)
    inst_map = ndi.label(mask)[0]
    return inst_map
    

def sauvola_thresh(prob_map, win_size=33):
    thresh = filters.threshold_sauvola(prob_map, window_size=win_size)
    mask = prob_map > thresh
    mask = cv2_opening(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = morph.remove_small_objects(mask, min_size=64)
    inst_map = ndi.label(mask)[0]
    return inst_map


def morph_chan_vese_thresh(prob_map):
    init_ls = segm.checkerboard_level_set(prob_map.shape, 2)
    ls = segm.morphological_chan_vese(prob_map, 35, smoothing=1, init_level_set=init_ls, lambda1=1, lambda2=1)
    hist = np.histogram(ls)[0]
    
    if hist[-1] > hist[0]:
        ls = 1 - ls
        
    ls = morph.remove_small_objects(ls.astype(bool), min_size=64)
    ls = ndi.binary_fill_holes(ls)
    inst_map =  ndi.label(ls)[0]
    return inst_map
  

def cv2_opening(mask):
    mask[mask > 0] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_mask = (mask*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    inst_map = ndi.label(new_mask)[0]
    return inst_map



def cv2_closing(mask):
    mask[mask > 0] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_mask = (mask*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    inst_map = ndi.label(new_mask)[0]
    return inst_map


def sobel_watershed(prob_map, inst_map, win_size=13):
    
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
    mask = segm.watershed(elevation_map, markers, mask=ann, watershed_line=True)
    
    # remove small objs. Will enhance PQ a lot '(HACKish)'
    mask[mask > 0] = 1
    mask = morph.remove_small_objects(mask.astype(bool), min_size=100)
    inst_map = ndi.label(mask)[0]
    
    return inst_map


# adapted from https://github.com/vqdang/hover_net/blob/master/src/postproc/other.py
def inv_dist_watershed(inst_map, win_size=13):
    
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
                    
    # watershed
    mask = segm.watershed(-distmap, markers, mask=ann, watershed_line=True)
    
    # remove small cells. Will enhance PQ a lot '(HACKish)'
    mask[mask > 0] = 1
    mask = morph.remove_small_objects(mask.astype(bool), min_size=100)
    inst_map = ndi.label(mask)[0]
    
    return inst_map


# Adapted from https://github.com/nicolefinnie/kaggle-dsb2018/blob/master/src/modules/image_processing.py
def random_walk(inst_map, contour):
    # THIS IS NOT WORKING AT ALL!
    # Morphological opening for smoothing the likely jagged edges
    # This typically improves PQ
    seg = np.copy(inst_map)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_contour = (contour*255).astype(np.uint8)
    seg[seg > 0] = 1
    new_mask = (seg*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    _, thresh_mask = cv2.threshold(new_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh_contour = cv2.threshold(new_contour, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
       
    sure_bg = cv2.dilate(thresh_mask, kernel, iterations=3)
    sure_fg = cv2.subtract(thresh_mask, thresh_contour)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=2)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = cv2.connectedComponentsWithStats(sure_fg)
    markers = markers[1]
    stats = markers[2]

    markers = markers + 1
    markers[unknown == 255] = 0
    labels = segm.random_walker(thresh_mask, markers)
    
    labels[labels==-1] = 0
    labels[labels==1] = 0
    labels = labels -1
    labels[labels==-1] = 0
    labels[labels > 0] = 1

    inst_map = ndi.label(labels)[0]
    return inst_map