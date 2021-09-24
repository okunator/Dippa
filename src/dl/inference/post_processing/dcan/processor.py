import numpy as np
from typing import Dict, Optional, Tuple, List

from ..base_processor import PostProcessor
from .post_proc import post_proc_dcan


class DCANPostProcessor(PostProcessor):
    def __init__(self,
                 thresh_method: str="naive",
                 thresh: float=0.5,
                 **kwargs) -> None:
        """
        Wrapper class for the DCAN post-processing pipeline for networks
        outputting contour maps. 

        https://arxiv.org/abs/1604.02677

         Args:
        ----------
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance
                branch. One of: "naive", "argmax", "sauvola", "niblack".
            thresh (float, default = 0.5): 
                threshold prob value. Used if `thresh_method` == "naive"
        """
        super(DCANPostProcessor, self).__init__(thresh_method, thresh)

    def post_proc_pipeline(self, maps: List[np.ndarray]) -> Tuple[np.ndarray]:
        """
        1. Run the dcan post-proc.
        2. Combine type map and instance map

        Args:
        -----------
            maps (List[np.ndarray]):
                A list of the name of the file, soft mask, and contour 
                map from the network
        """
        name = maps[0]
        prob_map = maps[1]
        contour_map = maps[2]
        type_map = maps[3]

        inst_map = post_proc_dcan(prob_map[..., 1], contour_map.squeeze())

        combined = None
        if type_map is not None:
            combined = self.combine_inst_type(inst_map, type_map)
        
        # Clean up the result
        inst_map = self.clean_up(inst_map)

        return name, inst_map, combined


    def run_post_processing(
            self,
            inst_probs: Dict[str, np.ndarray],
            aux_maps: Dict[str, np.ndarray],
            type_probs: Dict[str, np.ndarray]
        ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Run post processing for all predictions

        Args:
        ----------
            inst_probs (Dict[str, np.ndarray]):
                Dictionary of (file name, soft instance map) pairs
                inst_map. Shape (H, W, 2) 
            aux_maps (Dict[str, np.ndarray]):
                Dictionary of (file name, dist map) pairs.
                The regressed contours from auxiliary branch. 
                Shape (H, W, 1)
            type_probs (Dict[str, np.ndarray]):
                Dict of (file name, type map) pairs. Type maps are in 
                one hot format (H, W, n_classes).

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
                aux_maps.values(), 
                type_probs.values()
            )
        )
        seg_results = self.parallel_pipeline(maps)
        
        return seg_results