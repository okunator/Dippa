import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from src.img_processing.process_utils import get_inst_centroid, bounding_box


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
    pred_id_list = list(np.unique(inst_map))[1:]
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