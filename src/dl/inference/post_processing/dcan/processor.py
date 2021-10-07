import numpy as np
from typing import Dict, Tuple, List

from ..base_processor import PostProcessor
from .post_proc import post_proc_dcan


class DCANPostProcessor(PostProcessor):
    def __init__(
        self,
        thresh_method: str="naive",
        thresh: float=0.5,
        **kwargs
    ) -> None:
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
        maps = self._threshold_probs(maps)
        maps["inst_map"] = post_proc_dcan(
            maps["inst_probs"][..., 1], maps["aux_map"].squeeze()
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
            aux_maps: Dict[str, np.ndarray],
        ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Run post processing for all predictions

        Args:
        ----------
            inst_probs (Dict[str, np.ndarray]):
                Dictionary of (file name, soft instance map) pairs
                inst_map. Shape (H, W, 2) 
            type_probs (Dict[str, np.ndarray]):
                Dict of (file name, type map) pairs. Type maps are in 
                one hot format (H, W, n_classes).
            sem_probs (Dict[str, np.ndarray]):
                Dictionary of (file name, sem map) pairs.
                sem maps are in one hot format (H, W, n_classes).
            aux_maps (Dict[str, np.ndarray]):
                Dictionary of (file name, dist map) pairs.
                The regressed contours from auxiliary branch. 
                Shape (H, W, 1)

        Returns:
        -----------
           List: A list of tuples containing filename (str), 
                  post-processed inst, aux, sem, and type map (ndarray).
                  
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
                sem_probs.values(),
                aux_maps.values(),
            )
        )
        seg_results = self._parallel_pipeline(maps)
        
        return seg_results