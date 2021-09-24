"""
MIT License

Copyright (c) 2020 vqdang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from src.utils import get_inst_centroid, bounding_box


# Adapted from https://github.com/vqdang/hover_net/blob/master/src/process.py
def combine_inst_semantic(
        inst_map: np.ndarray,
        type_map: np.ndarray
    ) -> np.ndarray:
    """
    Takes in the outputs of the different segmentation heads and 
    combines them into one instance segmentation result

    Args:
    -----------
        inst_map (np.ndarray): 
            the post processed output from the instance segmentation 
            head of the model. Shape (H, W).
        type_map (np.ndarray): 
            output from the type segmentation head of the model. 
            Shape (H, W).

    Returns:
    -----------
        np.ndarray: Resulting type map with nuclei separated. 
        Shape (H, W).
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


    # If inst_type == "bg" (instance has no label) find the nearest 
    # instance with a class label and use that. This way no instances 
    # are dropped from the end result 
    inst_df = pd.DataFrame(inst_ids).transpose()
    type_map_out = np.zeros_like(type_map)

    # Skip imgs w/o any predicted nuclei
    if not inst_df.empty:
        cm = inst_df[["centroid_x", "centroid_y", "inst_type"]]
        found_centroids = cm[cm["inst_type"] != "bg"]
        missed_centroids = cm[cm["inst_type"] == "bg"]
        mc = missed_centroids[["centroid_x", "centroid_y"]].values
        fc = found_centroids[["centroid_x", "centroid_y"]].values
        
        # Skip imgs with less than one found centroid
        if fc.shape[0] > 1:
            centroid_dists = distance_matrix(mc, fc)
            closest_centroids = fc[np.argmin(centroid_dists, axis=1)]

            # closest centroids in df
            cc_df = pd.DataFrame(
                closest_centroids,
                columns=["centroid_x", "centroid_y"]
            )
            cc_df = found_centroids.merge(
                cc_df,
                on=["centroid_x", "centroid_y"],
                how="inner"
            )

            # cells that did not have a class assigned
            missed_centroids = missed_centroids.assign(
                inst_type=cc_df["inst_type"].values
            )

            inst_df = inst_df.merge(
                missed_centroids,
                on=["centroid_x", "centroid_y"],
                how="left"
            )
            inst_df.index += 1
            inst_ids = inst_df.transpose().to_dict()

            for inst_id, val in inst_ids.items():
                inst_tmp = inst_map == inst_id
                inst_type = type_map[inst_map == inst_id].astype("uint16")

                inst_type = val["inst_type_x"]
                if val["inst_type_x"] == "bg":
                    inst_type = val["inst_type_y"]

                type_map_out += (inst_tmp * int(inst_type))

    return type_map_out