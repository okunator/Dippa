import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, List

from .post_proc import post_proc_cellpose
from ..base_processor import PostProcessor


class CellposePostProcessor(PostProcessor):
    def __init__(
            self,
            thresh_method: str="naive",
            thresh: float=0.5,
            **kwargs
        ) -> None:
        """
        Wrapper class to run the CellPose post processing pipeline for 
        networks outputting instance maps, Optional[type maps], 
        and horizontal & vertical maps.

        CellPose:
        https://www.nature.com/articles/s41592-020-01018-x

        Args:
        -----------
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance
                branch. One of: "naive", "argmax", "sauvola", "niblack".
            thresh (float, default = 0.5): 
                threshold prob value. Used if `thresh_method` == "naive"
        """
        super(CellposePostProcessor, self).__init__(thresh_method, thresh)
        self.flows = OrderedDict()

    def post_proc_pipeline(
            self,
            maps: List[np.ndarray]
        ) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        1. Threshold
        2. Post process instance map
        3. Combine type map and instance map

        Args:
        -----------
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks, and hover
                maps from the network

        Returns:
        ----------
            Tuple: the filename (str), instance segmentation mask (H, W)
            and semantic segmentation mask (H, W).
        """
        maps = self._threshold_probs(maps)
        cellpose_dict = post_proc_cellpose(maps["aux_map"], maps["inst_map"])
        maps["inst_map"] = cellpose_dict["inst_map"]
        maps["inst_map"], maps["type_map"] = self._finalize_inst_seg(maps)

        res = [
            map for key, map in maps.items() 
            if not any([l in key for l in ("probs", "aux")])
        ]

        # save the flows here to avoid complicating the inferer code
        self.flows[maps["fn"]] = cellpose_dict["flows"]["flow"]

        return res

    def run_post_processing(
            self,
            inst_probs: Dict[str, np.ndarray],
            type_probs: Dict[str, np.ndarray],
            sem_probs: Dict[str, np.ndarray],
            aux_maps: Dict[str, np.ndarray],
        ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Run post processing for all predictions

        Args:
        ------------
            inst_probs (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, soft instance map) pairs
                inst_map shapes are (H, W, 2) 
            type_probs (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, type map) pairs.
                type maps are in one hot format (H, W, n_classes).
            sem_probs (Dict[str, np.ndarray]):
                Dictionary of (file name, sem map) pairs.
                sem maps are in one hot format (H, W, n_classes).
            aux_maps (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, hover map) pairs.
                hover_map[..., 0] = horizontal map
                hover_map[..., 1] = vertical map

        Returns:
        -----------
            List: a list of tuples containing filename, post-processed 
            inst map and type map
            
            Example: 
            [("filename1", inst_map: np.ndarray, type_map: np.ndarray),
             ("filename2", inst_map: np.ndarray, type_map: np.ndarray)]
        """
        # Set arguments for threading pool
        maps = list(
            zip(
                inst_probs.keys(), 
                inst_probs.values(), 
                type_probs.values(),
                sem_probs.values(),
                aux_maps.values(), 
            )
        )
        seg_results = self._parallel_pipeline(maps)
        
        return seg_results