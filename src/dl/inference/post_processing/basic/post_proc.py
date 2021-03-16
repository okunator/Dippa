import cv2
import numpy as np
import skimage.morphology as morph
import skimage.segmentation as segm
import skimage.feature as feat
import skimage.filters as filters
import skimage.util as util
import scipy.ndimage as ndi

from ..thresholding import niblack_thresh

from src.utils.mask_utils import (
    bounding_box, 
    remap_label,
    remove_debris, 
    cv2_opening, 
    binarize,
    remove_small_objects
)


def shape_index_watershed2(prob_map: np.ndarray,
                           inst_map: np.ndarray,
                           sigma: float = 3.0,
                           **kwargs) -> np.ndarray:
    """
    After thresholding, this function can be used to post process each nuclei instance.
    This uses shape index to find local curvature from the probability map and uses that information
    to separate overlapping nuclei. This is an updated version of 'shape_index_watershed' where closing
    operations remove a lot small cells.

    Args:
    ----------
        prob_map (np.ndarray): 
            The soft mask outputted from the network. Shape (H, W)
        inst_map (np.ndarray): 
            The instance map to be segmented. Shape (H, W)
        sigma (float, default=3.0): 
            Std for gaussian kernel before computing shape index

    Returns:
    ----------
        np.ndarray: post-processed labelled inst_map
    """
    s = feat.shape_index(prob_map, sigma=sigma)
    
    # Spherical cap
    s[s > 1] = 1
    s[s <= 1] = 0
    s = ndi.binary_fill_holes(np.nan_to_num(s*inst_map))
    s = remove_small_objects(s.astype(bool), 8, connectivity=1)
    s = ndi.label(s)[0]

    shape = s.shape[:2]
    nuc_list = list(np.unique(s))
    nuc_list.remove(0)

    mask = np.zeros(shape, dtype=np.int32)
    markers = np.zeros(shape, dtype=np.int32)
    for nuc_id in nuc_list:

        nuc_map = np.copy(s == nuc_id)
        y1, y2, x1, x2 = bounding_box(nuc_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        nuc_map_crop = nuc_map[y1:y2, x1:x2].astype("int32")
        marker_crop = np.copy(nuc_map_crop)

        # Erode to get markers
        marker = morph.binary_erosion(binarize(marker_crop))
        marker = morph.binary_erosion(marker)
        marker = morph.binary_erosion(marker)
        marker = morph.binary_opening(marker)

        # if the erosion removed the whole label
        # just use the original as a marker
        nuc_id_map = ndi.label(marker)[0]
        nuc_ids = np.unique(nuc_id_map)
        if len(nuc_ids) == 1:
            nuc_id_map = binarize(nuc_map_crop)
            nuc_ids = np.unique(nuc_id_map)

        # relabel the new instances after erosion
        # starting from the max id + 1 of the already
        # iterated nucelis.
        new_nucs = nuc_ids[1:]
        num_new_nucs = len(new_nucs)
        nuc_ids_so_far = np.unique(markers)
        start = np.max(nuc_ids_so_far) + 1
        end = start + num_new_nucs
        if end - start == 1:
            new_nuc_ids = np.array([start])
        else:
            new_nuc_ids = np.arange(start, end)

        for old_id, new_id in list(zip(new_nucs, new_nuc_ids)):
            nuc_id_map[nuc_id_map == old_id] = new_id

        markers[y1:y2, x1:x2] = nuc_id_map

        # compute distance map from the marker
        distance = ndi.distance_transform_edt(nuc_id_map)
        distance = 255 * (distance / np.amax(distance))
        distance = cv2.GaussianBlur(ndi.filters.maximum_filter(distance, 7), (3, 3), 0)

        ws_temp = segm.watershed(-distance, mask=nuc_map_crop, markers=nuc_id_map, watershed_line=True)

        id_count = 1
        cell_ids = np.unique(ws_temp)[1:]
        mask_new = np.zeros(ws_temp.shape[:2], dtype=np.int32)
        for sub_nuc_id in cell_ids:
            sub_mask = np.copy(ws_temp == sub_nuc_id).astype("int32")

            # Gets rid of small stuff
            sub_mask = filters.rank.median(util.img_as_ubyte(sub_mask), morph.disk(3))

            # Dilate
            sub_mask = morph.binary_dilation(binarize(sub_mask))

            # Fill holes
            sub_mask = ndi.binary_fill_holes(sub_mask)

            sub_mask_inst = sub_mask*sub_nuc_id
            mask_new += sub_mask_inst

            # if cells end up overlapping after dilations then remove the overlaps
            # so no new ids are created when summing overlapping ids to the result mask
            new_ids = np.unique(mask_new)[1:]
            if id_count < len(new_ids):
                for ix in new_ids[int(np.where(new_ids == sub_nuc_id)[0]+1):]:
                    mask_new[mask_new == ix] = 0
            id_count += 1

        mask[y1:y2, x1:x2] += mask_new

    inst_map = remove_debris(mask, 18)
    inst_map = remap_label(mask)
    return inst_map


# Explorative
def shape_index_watershed(prob_map: np.ndarray,
                          inst_map: np.ndarray,
                          win_size: int = 13,
                          **kwargs) -> np.ndarray:
    """
    After thresholding, this function can be used to post process each nuclei instance.
    This uses shape index to find local curvature from the probability map and uses that information
    to separate overlapping nuclei before the watershed segmentatiton. Markers for the
    distance maps are computed using niblack thresholding on the distance map.

    Args:
    ----------
        prob_map (np.ndarray): 
            The soft mask outputted from the network. Shape (H, W)
        inst_map (np.ndarray): 
            The instance map to be segmented. Shape (H, W)
        win_size (int, default=13): 
            window size used in niblack thresholding the distance maps to
            find markers for watershed

    Returns:
    -----------
        np.ndarray: post-processed labelled inst_map
    """
    shape = inst_map.shape[:2]
    nuc_list = list(np.unique(inst_map))
    nuc_list.remove(0)

    # find the distance map per nuclei instance
    distmap = np.zeros(shape, dtype=np.uint8)
    for nuc_id in nuc_list:
        nuc_map = np.copy(inst_map == nuc_id)

        # Do operations to the bounded box of the nuclei
        # rather than the full size matrix
        y1, y2, x1, x2 = bounding_box(nuc_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        nuc_map_crop = nuc_map[y1:y2, x1:x2]

        # Find distance transform of the bounding box and normalize it
        distance = ndi.distance_transform_edt(nuc_map_crop)
        distance = 255 * (distance / np.amax(distance))
        distmap[y1:y2, x1:x2] += distance.astype('uint8')

    # find markers for ws
    markers = np.copy(distmap)
    markers = niblack_thresh(distmap, win_size)

    # Find local curvature with shape index
    # use it to separate clumps
    s = feat.shape_index(prob_map)
    s[s > 0] = 1
    s[s <= 0] = 0
    s = ndi.binary_fill_holes(s)
    mask = ndi.label(s*inst_map)[0]
    mask = cv2_opening(mask, iterations=2)

    mask = segm.watershed(-distmap, markers, mask=mask, watershed_line=True)
    mask[mask > 0] = 1
    mask = ndi.binary_fill_holes(mask)
    inst_map = ndi.label(mask)[0]

    return inst_map


def sobel_watershed(prob_map: np.ndarray,
                    inst_map: np.ndarray,
                    win_size: int = 13,
                    **kwargs) -> np.ndarray:
    """
    After thresholding, this function can be used to compute distance maps for each nuclei instance
    and watershed segment the elevation map of the prob_map (sobel). Before computing distance maps
    a binary opening is performed to the instance map. Markers for the distance maps are computed
    using niblack thresholding on the distance map.

    Args:
    ----------
        prob_map (np.ndarray): 
            The soft mask outputted from the network
        inst_map (np.ndarray): 
            The instance map to be segmented. Shape (H, W)
        win_size (int, default=13): 
            window size used in niblack thresholding the distance maps to
            find markers for watershed

    Returns:
    ----------
        np.ndarray: post-processed labelled inst_map
    """

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
    
    return inst_map


# adapted from https://github.com/vqdang/hover_net/blob/master/src/postproc/other.py
def inv_dist_watershed(inst_map: np.ndarray, win_size: int = 13, **kwargs) -> np.ndarray:
    """
    After thresholding, this function can be used to compute distance maps for each nuclei instance
    and watershed segment the inverse distmaps. Before computing distance maps a binary opening
    is performed to the instance map. Markers for the distance maps are computed using 
    niblack thresholding on the distance map.

    Args:
    ----------
        inst_map (np.ndarray): 
            The instance map to be segmented
        win_size (int, default=13): 
            window size used in niblack thresholding the 
            distance maps to find markers for watershed

    Returns:
    ----------
        np.ndarray: post-processed labelled inst_map
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
                    
    inst_map = segm.watershed(-distmap, markers, mask=ann, watershed_line=True)
        
    return inst_map