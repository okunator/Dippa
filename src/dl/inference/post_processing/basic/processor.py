import numpy as np
from typing import Dict, Tuple, List

from .post_proc import shape_index_watershed2
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
        name = maps[0]
        prob_map = maps[1]
        type_map = maps[2]

        inst_map = self.threshold(prob_map)
        inst_map = shape_index_watershed2(prob_map[..., 1], inst_map)

        combined = None
        if type_map is not None:
            combined = self.combine_inst_type(inst_map, type_map)
        
        # Clean up the result
        inst_map = self.clean_up(inst_map)

        return name, inst_map, combined

    def run_post_processing(
            self,
            inst_probs: Dict[str, np.ndarray],
            type_probs: Dict[str, np.ndarray],
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
            List: a list of tuples containing filename, post-processed 
            inst map and type map
            
            Example:
            [("filename1", inst_map: np.ndarray, type_map: np.ndarray)
             ("filename2", inst_map: np.ndarray, type_map: np.ndarray)]
        """
        # Set arguments for threading pool
        maps = list(
            zip(inst_probs.keys(), inst_probs.values(), type_probs.values())
        )
        seg_results = self.parallel_pipeline(maps)

        return seg_results