import numpy as np
from typing import Dict, Tuple, List

from .post_proc import shape_index_watershed
from ..base_processor import PostProcessor


class BasicPostProcessor(PostProcessor):
    def __init__(
            self,
            thresh_method: str="naive",
            thresh: float=0.5,
            **kwargs
        ) -> None:
        """
        Wrapper class to run the baseline basic post processing pipeline
        for networks outputting instance maps but no auxiliary maps

        Args:
        -----------
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance
                branch. One of: "naive", "argmax", "sauvola", "niblack".
            thresh (float, default = 0.5): 
                threshold prob value. Used if `thresh_method` == "naive"
        """
        super(BasicPostProcessor, self).__init__(thresh_method, thresh)

    def post_proc_pipeline(
            self,
            maps: List[np.ndarray]
        ) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        1. Threshold
        2. Post process instance map
        3. Combine type map and instance map

        Args:
        ----------
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks, and hover 
                maps from the network

        Returns:
        ----------
            Tuple: The filename (str), instance segmentation mask (H, W)
            and semantic segmentation mask (H, W).
        """
        maps = self._threshold_probs(maps)
        maps["inst_map"] = shape_index_watershed(
            maps["inst_probs"][..., 1], maps["inst_map"]
        )
        maps["inst_map"], maps["type_map"] = self._finalize_inst_seg(maps)

        res = [
            map for key, map in maps.items()
            if not any([l in key for l in ("probs", "aux")])
        ]

        return res

    def run_post_processing(
            self,
            inst_probs: Dict[str, np.ndarray],
            type_probs: Dict[str, np.ndarray],
            sem_probs: Dict[str, np.ndarray],
            **kwargs) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Run post processing for all predictions

        Args:
        ----------
            inst_probs (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, soft instance map) pairs
                inst_map shapes are (H, W, 2). Probability map
            type_probs (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, tsoft ype map) pairs.
                soft type maps are in one hot format (H, W, n_classes).

        Returns:
        -----------
           List: A list of tuples containing filename (str), 
                  post-processed inst, sem, and type map (ndarray).
                  
                  The output maps depend on the outputs of the network.
                  If the network does not output type or sem maps,
                  these are not contained in the result list.
            
            Output example:
            [("filename1", aux_map, inst_map, type_map, sem_map),
             ("filename2", aux_map, inst_map, type_map, sem_map)]
        """
        # Set arguments for threading pool
        maps = list(
            zip(
                inst_probs.keys(),
                inst_probs.values(),
                type_probs.values(),
                sem_probs.values()
            )
        )
        seg_results = self._parallel_pipeline(maps)

        return seg_results