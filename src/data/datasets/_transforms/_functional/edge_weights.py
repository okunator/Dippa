import numpy as np
from scipy.ndimage.morphology import distance_transform_edt


__all__ = ["gen_weight_map"]


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def gen_weight_maps(
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