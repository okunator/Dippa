"""
Most of the functions are from: https://github.com/vqdang/hover_net/blob/master/src/metrics/stats_utils.py
Some very minor naming changes done and comments added
"""

import numpy as np
import cv2

from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.metrics import variation_of_information

# Ported from here: https://github.com/vqdang/hover_net/blob/master/src/metrics/stats_utils.py
def AJI(true, pred):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result. 
    """
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    
    true_masks = {}
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.int8)
        true_masks[t] = t_mask
    
    pred_masks = {}
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.int8)
        pred_masks[p] = p_mask

    pairwise_inter = np.zeros([len(true_masks.keys())+1, 
                               len(pred_masks.keys())+1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_masks.keys())+1, 
                               len(pred_masks.keys())+1], dtype=np.float64)
    
    # for true_id in true_id_list[1:]: # 0-th is background
    for true_id in true_masks.keys():
        t_mask = true_masks[true_id]

        pred_true_overlap = pred[t_mask > 0] 
        pred_true_overlap_id = list(np.unique(pred_true_overlap))

        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore background
                continue

            p_mask = pred_masks[pred_id]
            intersection = t_mask*p_mask # True positives
            subtraction = t_mask-p_mask
            add = t_mask+p_mask

            TP = len(intersection[intersection == 1])
            TN = len(subtraction[subtraction == 0])
            FP = len(subtraction[subtraction == -1])
            FN = len(subtraction[subtraction == 1])
            total = FP + TP + FN # everything except background

            pairwise_inter[true_id-1, pred_id-1] = TP
            pairwise_union[true_id-1, pred_id-1] = total

    # Calculate the pairwise (matching nuclei pair) IoU    
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

    # Get the instance idxs of predicted nucleis that give highest IoU for each ground truth nuclei, 
    # dont care about reusing the prediction instance multiple times here
    paired_pred = np.argmax(pairwise_iou, axis=1)

    # Now get the corresponding IoU values 
    pairwise_iou = np.max(pairwise_iou, axis=1)

    # Exlude those that dont have intersection i.e. pairwise iou > 0 or intersection > 0
    # These are the indexes for the non-zero values in pairwise_iou var
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]

    # Get the prediction idxs that match with the indexes that actually have overlap (intersection > 0)
    # There may be lonely predction nucleis that don't have a matching ground truth nuclei or vice versa.
    # So basically these are only the true positive matches. False negatives and false positives excluded
    paired_pred = paired_pred[paired_true]

    # Calculate the overall intersection and union on these matching nucleis
    # NOTE: We have excluded the predictions of nuclei instances that are FALSE POSITIVES
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = (list(paired_true + 1))
    paired_pred = (list(paired_pred + 1))

    # Find the indices that were those FALSE POSITIVES and FALSE NEGATIVES
    # that were excluded then we add all those unpaired ground truths and predictions into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])

    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    # Calculate the final score
    aji_score = overall_inter / overall_union

    return aji_score


# ported from https://github.com/vqdang/hover_net/blob/master/src/metrics/stats_utils.py
def AJI_plus(true, pred):
    """
    AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI 
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.
    """        
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    pairwise_inter = np.zeros([len(true_id_list)-1, 
                               len(pred_id_list)-1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list)-1, 
                               len(pred_id_list)-1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]: # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id-1, pred_id-1] = inter
            pairwise_union[true_id-1, pred_id-1] = total - inter
    
    ######################################################################
    ######################################################################
    ######################################################################
    # AJI+ deviates here from AJI. It calculates unique pairings of nucleis
    # with the Hungarian algorithm that should maximize the prob
    # of overlapping nucleis being the actual unique matching
    
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair 
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = (list(paired_true + 1)) # index to instance ID
    paired_pred = (list(paired_pred + 1))
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    
    aji_score = overall_inter / overall_union
    return aji_score


# ported from https://github.com/vqdang/hover_net/blob/master/src/metrics/stats_utils.py
def DICE2(true, pred):
    """
    Ensemble dice.
    """
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
        
    overall_total = 0
    overall_inter = 0

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
                    
        for pred_id in pred_true_overlap_id:
            # ignore overlapping background
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / (overall_total + 1.0e-6)


# ported from https://github.com/vqdang/hover_net/blob/master/src/metrics/stats_utils.py
def PQ(true, pred, match_iou=0.4):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"
    
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
        
    pairwise_iou = np.zeros([len(true_id_list) -1, 
                             len(pred_id_list) -1], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]: # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlapping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id-1, pred_id-1] = iou
    
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1 # index is instance id - 1
        paired_pred += 1 # hence return back to original
    else:  
        # Exhaustive maximal unique pairing
        # Munkres pairing with scipy library (Hungarian algorithm)
        # the algorithm returns (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence 
        # is returned, thus the unique pairing is ensured
        # inverse pair to get high IoU as minimum   
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        # extract the paired cost and remove invalid pairs 
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    TP = len(paired_true)
    FP = len(unpaired_pred)
    FN = len(unpaired_true)

    recall = TP/(TP+FN+1.0e-6)
    precision = TP/(TP+FP+1.0e-6)

    # get the F1-score i.e DQ
    dq = TP / (TP + 0.5 * FP + 0.5 * FN + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (TP + 1.0e-6)
    pq = dq * sq
    
    res = {}
    res['pq'] = pq
    res['sq'] = sq
    res['dq'] = dq
    res['recall'] = recall
    res['precision'] = precision
    return res


def conventional_metrics(true, pred):
    """
    DICE = 2TP/(2TP + FP + FN)
    JACCARD = TP/(TP+FP+FN) or just DICE/(2-DICE) also called intersection over union IoU
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    """
    true[true > 0] = 1
    pred[pred > 0] = 1 
    intersection = true*pred # True positives
    subtract = true - pred
    
    TP = len(intersection[intersection == 1])
    TN = len(subtract[subtract == 0])
    FP = len(subtract[subtract == -1])
    FN = len(subtract[subtract == 1])
    DICE1 = 2*TP/(2*TP+FP+FN)
    JACCARD = DICE1/(2-DICE1) # IoU
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    return (DICE1, JACCARD, sensitivity, specificity)


def split_and_merge(true, pred):
    """
    Split and merge for all the predicted nuclei instances vs gt nuclei instances
    
    """
    return variation_of_information(true, pred)