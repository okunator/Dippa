import numpy as np
from typing import Dict, Tuple, List, Union

from .post_proc import post_proc_hover
from ..base_processor import PostProcessor


class HoverNetPostProcessor(PostProcessor):
    def __init__(
            self,
            thresh_method: str="naive",
            thresh: float=0.5,
            **kwargs
        ) -> None:
        """
        Wrapper class to run the HoVer-Net post processing pipeline for
        networks outputting instance maps, Optional[type maps], and 
        horizontal & vertical maps.        

        Args:
        ---------
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance
                branch. One of: "naive", "argmax", "sauvola", "niblack".
            thresh (float, default = 0.5): 
                threshold prob value. Used if `thresh_method` == "naive"
        """
        super(HoverNetPostProcessor, self).__init__(thresh_method, thresh)

    def post_proc_pipeline(self, maps: List[np.ndarray]) -> List[np.ndarray]:
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
        -----------
            List: list of the filename and different output masks
                  masks order: "inst", "types", "sem"
        """
        maps = self._threshold_probs(maps)
        maps["inst_map"] = post_proc_hover(maps["inst_map"], maps["aux_map"])
        maps["inst_map"], maps["type_map"] = self._finalize_inst_seg(maps)

        res = [
            map for key, map in maps.items() 
            if not any([l in key for l in ("probs", "aux")])
        ]

        return res

    def run_post_processing(
            self,
            inst_probs: Dict[str, np.ndarray],
            type_probs: Dict[str, Union[np.ndarray, None]],
            sem_probs: Dict[str, Union[np.ndarray, None]],
            aux_maps: Dict[str, np.ndarray],
        ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Run post processing for all predictions

        Args:
        ---------
            inst_probs (Dict[str, np.ndarray]):
                Dictionary of (file name, soft instance map) pairs
                inst_map shapes are (H, W, 2) 
            type_probs (Dict[str, np.ndarray | None]):
                Dictionary of (file name, type map) pairs.
                type maps are in one hot format (H, W, n_classes).
            sem_probs (Dict[str, np.ndarray | None]):
                Dictionary of (file name, sem map) pairs.
                sem maps are in one hot format (H, W, n_classes).
            aux_maps (Dict[str, np.ndarray]):
                Dictionary of (file name, hover map) pairs.
                hover_map[..., 0] = horizontal map
                hover_map[..., 1] = vertical map

        Returns:
        -----------
            List: a list of tuples containing filename, post-processed 
            inst map and type map
            
            Example: 
            [("filename1", inst_map: np.ndarray, aux_map: np.ndarray),
             ("filename2", inst_map: np.ndarray, aux_map: np.ndarray)]
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