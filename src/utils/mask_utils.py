import cv2
import numpy as np
from typing import List
from scipy import ndimage as ndi
from skimage import morphology as morph
from scipy.ndimage.morphology import distance_transform_edt


# From https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/misc.py
# warning removed
def remove_small_objects(
        ar: np.ndarray, 
        min_size: int=64,
        connectivity: int=1,
        in_place: bool=False,
        *, 
        out: np.ndarray=None):
    """Remove objects smaller than the specified size.
    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type 
        is int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used 
        during labelling if `ar` is bool.
    in_place : bool, optional (default: False)
        If ``True``, remove the objects in the input array itself.
        Otherwise, make a copy. Deprecated since version 0.19. Please
        use `out` instead.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.
    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or 
        string.
    ValueError
        If the input array contains negative values.
    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.
    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True
    """

    if out is not None:
        in_place = False

    if in_place:
        out = ar
    elif out is None:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def binarize(inst_map: np.ndarray) -> np.ndarray:
    """
    Binarize a labelled instance map

    Args:
    ----------
        inst_map (np.ndarray): 
            Instance map to be binarized

    Returns:
    -----------
        np.ndarray: Binary mask. Shape (H, W).
    """
    binary = np.copy(inst_map > 0)
    return binary.astype("uint8")


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def fix_duplicates(inst_map: np.ndarray) -> np.ndarray:
    """
    Deal with duplicated instances in an inst map. For example, 
    duplicated instances due to mirror padding. 

    Args:
    -----------
        inst_map (np.ndarray): Inst map

    Returns:
    -----------
        np.ndarray: The instance segmentation map without duplicated 
        indices. Shape (H, W). 
    """
    current_max_id = np.amax(inst_map)
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0) # 0 is background
    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        remapped_ids = ndi.label(inst)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        inst_map[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(inst_map)

    return inst_map


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def remove_1px_boundary(inst_map: np.ndarray) -> np.ndarray:
    """
    Removes 1px around each instance, removing overlaps of cells in an 
    inst map

    Args: 
    ----------
        inst_map (np.ndarray): 
            instance map

    Returns:
    -----------
        np.ndarray: The instance segmentation map with 1px of instance 
        boundaries removed. Shape (H, W).
    """
    new_inst_map = np.zeros(inst_map.shape[:2], np.int32)
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0) # 0 is background
    k = morph.disk(1)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        inst = cv2.erode(inst, k, iterations=1)
        new_inst_map[inst > 0] = inst_id

    return new_inst_map


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def get_weight_map(
        inst_map: np.ndarray,
        sigma: float=5.0,
        w0: float=10.0
    ) -> np.ndarray:
    """
    Generate a weight map like in U-Net paper

    Args: 
    -----------
        inst_map (np.ndarray): 
            Instance map
        sigma (float): 
            Factor multiplied to the for the distance maps
        w0 (float): 
            Weight multiplied to the penalty map 

    Returns:
    -----------
        np.ndarray: Nuclei boundary weight map. Shape (H, W).
    """
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0) # 0 is background

    if len(inst_list) <= 1: # 1 instance only
        return np.zeros(inst_map.shape[:2])
    stacked_inst_bgd_dst = np.zeros(inst_map.shape[:2] + (len(inst_list),))

    for idx, inst_id in enumerate(inst_list):
        inst_bgd_map = np.array(inst_map != inst_id , np.uint8)
        inst_bgd_dst = distance_transform_edt(inst_bgd_map)
        stacked_inst_bgd_dst[..., idx] = inst_bgd_dst

    near1_dst = np.amin(stacked_inst_bgd_dst, axis=2)
    near2_dst = np.expand_dims(near1_dst, axis=2)
    near2_dst = stacked_inst_bgd_dst - near2_dst
    near2_dst[near2_dst == 0] = np.PINF # very large
    near2_dst = np.amin(near2_dst, axis=2)
    near2_dst[inst_map > 0] = 0 # the instances
    near2_dst = near2_dst + near1_dst
    # to fix pixel where near1 == near2
    near2_eve = np.expand_dims(near1_dst, axis=2)
    # to avoide the warning of a / 0
    near2_eve = (1.0 + stacked_inst_bgd_dst) / (1.0 + near2_eve)
    near2_eve[near2_eve != 1] = 0
    near2_eve = np.sum(near2_eve, axis=2)
    near2_dst[near2_eve > 1] = near1_dst[near2_eve > 1]
    #
    pix_dst = near1_dst + near2_dst
    pen_map = pix_dst / sigma
    pen_map = w0 * np.exp(- pen_map**2 / 2)
    pen_map[inst_map > 0] = 0 # inner instances zero

    return pen_map


def center_crop(img: np.ndarray, ch: int, cw: int) -> np.ndarray:
    """
    Center crop an input image

    Args:
    ----------
        img (np.ndarray): 
            Input img. Shape (H, W).
        ch (int):
            Crop height
        cw (int):
            crop width

    Returns:
    ----------
        np.ndarray: Center cropped image. Shape (ch, cw).
    """
    if len(img.shape) == 3:
        H, W, _ = img.shape
    else:
        H, W = img.shape

    x = W // 2 - (cw // 2)
    y = H // 2 - (ch // 2)    

    if len(img.shape) == 3:
        img = img[y:y + ch, x:x + cw, :]
    else:
        img = img[y:y + ch, x:x + cw]

    return img


# Ported from https://github.com/vqdang/hover_net/blob/master/src/misc/utils.py
def bounding_box(inst_map: np.ndarray) -> List[int]:
    """
    Bounding box coordinates for nuclei instance
    that is given as input. This assumes that the inst_map 
    has only one instance in it.

    Args:
    ----------
        inst_map (np.ndarray): 
            Instance labels

    Returns:
    ----------
        List: List of the origin- and end-point coordinates of the bbox
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
def remap_label(pred: np.ndarray) -> np.ndarray:
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be 
    reordered so that bigger nucler has smaller ID

    Args:
    -----------
        pred (np.ndarray):
            The 2d array contain instances where each instances is
            marked by non-zero integer
    
    Returns:
    -----------
        np.ndarray: inst map with remapped contiguous labels
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1

    return new_pred


# Ported from https://github.com/vqdang/hover_net/blob/master/src/misc/utils.py
def get_inst_centroid(inst_map: np.ndarray) -> np.ndarray:
    """
    Get centroid x, y coordinates from each unique nuclei instance

    Args:
    ----------
        inst_map (np.ndarray): 
            Nuclei instance map

    Returns:
    ----------
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


def get_inst_types(
        inst_map: np.ndarray, 
        type_map: np.ndarray
    ) -> np.ndarray:
    """
    Get the types of every single instance in an instance map
    and write them to a 1D-Array
    
    Args:
    ----------
        inst_map (np.ndarray): 
            Instance map of shape (H, W)
        type_map (np.ndarray): 
            Type map of shape (H, W). Labels are indices.

    Returns:
    ----------
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


def get_type_instances(
        inst_map: np.ndarray,
        type_map: np.ndarray,
        class_num: int
    ) -> np.ndarray:
    """
    Get the instances from an instance map that belong to class 
    'class_num' Drop everything else. The type map and inst map need to
    have the exact same non-zero pixels.
    
    Args:
    ----------
        inst_map (np.ndarray): 
            Instance map of shape (H, W)
        type_map (np.ndarray): 
            Type map of shape (H, W). Labels are indices.
        class_num (int): 
            Class label
    
    Returns:
    ----------
        np.ndarray: Numpy ndarray  of shape (H, W) where the values 
        equalling 'class_num' are dropped
    """
    t = type_map.astype("uint8") == class_num
    imap = np.copy(inst_map)
    imap[~t] = 0

    return imap


def one_hot(type_map: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert type map of shape (H, W) to one hot encoded types of shape: 
    (H, W, C)
    
    Args:
    -----------
        type_map (np.ndarray): 
            Type map of shape (H, W). Labels are indices.
        num_classes (int): 
            Number of classes in the dataset

    Returns:
    -----------
        np.ndarray: Numpy ndarray of the input array (H, W) in one hot 
        format. Shape: (H, W, num_classes).
    """
    return np.eye(num_classes+1)[type_map]


def type_map_flatten(type_map: np.ndarray) -> np.ndarray:
    """
    Convert a one hot type map of shape (H, W, C) to a single channel
    indice map of shape (H, W)
    
    Args:
    -----------
        type_map (np.ndarray): 
            Type map to be flattened

    Returns
    -----------
        np.ndarray: Flattened one hot np.ndarray. 
        I.e. (H, W, C) --> (H, W)
    """
    type_out = np.zeros([type_map.shape[0], type_map.shape[1]])
    for t in np.unique(type_map):
        type_tmp = type_map == t
        type_out += (type_tmp * t)

    return type_out


def to_inst_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Takes in a binary mask -> fill holes -> removes small objects -> 
    label connected components. If class channel is included this 
    assumes that binary_mask[..., 0] is the bg channel and 
    binary_mask[..., 1] the foreground.

    Args:
    -----------
        binary_mask (np.ndarray): 
            A binary mask to be labelled. Shape (H, W) or (H, W, C)
    
    Returns:
    -----------
        np.ndarray: labelled instances np.ndarray of shape (H, W)
    """
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[..., 1]

    mask = ndi.binary_fill_holes(binary_mask)
    mask = remove_small_objects(binary_mask.astype(bool), min_size=10)
    inst_map = ndi.label(mask)[0]

    return inst_map


def cv2_opening(
        inst_map: np.ndarray,
        iterations: int=2
    ) -> np.ndarray:
    """
    Takes in an inst_map -> binarize -> apply morphological opening 
    (2 iterations) -> label

    Args:
    -----------
        inst_map (np.ndarray): 
            Instance map to be opened. Shape (H, W)
        iterations (int, default=2):
            Number of iterations for the operation

    Returns:
    -----------
        np.ndarray: Morphologically opened np.ndarray of shape (H, W)
    """
    inst_map = binarize(inst_map)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_inst_map = (inst_map*255).astype(np.uint8)
    new_inst_map = cv2.morphologyEx(
        new_inst_map, cv2.MORPH_OPEN,
        kernel, iterations=iterations
    )
    inst_map = ndi.label(new_inst_map)[0]

    return inst_map


def cv2_closing(
        inst_map: np.ndarray,
        iterations: int=2
    ) -> np.ndarray:
    """
    Takes in an inst_map -> binarize -> apply morphological closing 
    (2 iterations) -> label
    
    Args:
    -----------
        inst_map (np.ndarray): 
            Instance map to be opened. Shape (H, W)
        iterations (int, default=2): 
            Number of iterations for the operation

    Returns:
    -----------
        np.ndarray: Morphologically closed np.ndarray of shape (H, W)
    """
    inst_map = binarize(inst_map)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    new_inst_map = (inst_map*255).astype(np.uint8)
    new_inst_map = cv2.morphologyEx(
        new_inst_map, cv2.MORPH_CLOSE,
        kernel, iterations=iterations
    )
    inst_map = ndi.label(new_inst_map)[0]

    return inst_map


def remove_debris(inst_map: np.ndarray, min_size: int = 10):
    """
    (Actually) Remove small objects from an inst map

    Args:
    ------------
        inst_map (np.ndarray): 
            Instance map
        min_size (int, default=10): 
            Min size for the objects that are left untouched

    Returns:
    -----------
        np.ndarray: Cleaned np.ndarray of shape (H, W)
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

        nuc_map_crop = remove_small_objects(
            nuc_map_crop.astype(bool), 
            min_size, connectivity=1
        ).astype("int32")

        nuc_map_crop[nuc_map_crop > 0] = ix
        res[y1:y2, x1:x2] += nuc_map_crop

    return res

