import numpy as np
from typing import Dict, Optional, Tuple, List

from .post_proc import post_proc_hover
from ..base_processor import PostProcessor


class HoverNetPostProcessor(PostProcessor):
    def __init__(self,
                 thresh_method: str="naive",
                 thresh: float=0.5,
                 **kwargs) -> None:
        """
        Wrapper class to run the HoVer-Net post processing pipeline for networks
        outputting instance maps, Optional[type maps], and horizontal & vertical maps.

        HoVer-Net:
        https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

        Args:
        ---------
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the insntance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
        """
        super(HoverNetPostProcessor, self).__init__(thresh_method, thresh)

    def post_proc_pipeline(self, maps: List[np.ndarray]) -> Tuple[np.ndarray]:
        """
        1. Threshold
        2. Post process instance map
        3. Combine type map and instance map

        Args:
        -----------
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks, and hover maps from the network
        """
        name = maps[0]
        prob_map = maps[1]
        hover_map = maps[2]
        type_map = maps[3]

        inst_map = self.threshold(prob_map)
        inst_map = post_proc_hover(inst_map, hover_map)

        combined = None
        if type_map is not None:
            combined = self.combine_inst_type(inst_map, type_map)
        
        # Clean up the result
        inst_map = self.clean_up(inst_map)

        return name, inst_map, combined

    def run_post_processing(self,
                            inst_probs: Dict[str, np.ndarray],
                            aux_maps: Dict[str, np.ndarray],
                            type_probs: Dict[str, np.ndarray]):
        """
        Run post processing for all predictions

        Args:
        ---------
            inst_probs (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, soft instance map) pairs
                inst_map shapes are (H, W, 2) 
            aux_maps (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, hover map) pairs.
                hover_map[..., 0] = horizontal map
                hover_map[..., 1] = vertical map
            type_probs (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, type map) pairs.
                type maps are in one hot format (H, W, n_classes).
        """
        # Set arguments for threading pool
        maps = list(zip(inst_probs.keys(), inst_probs.values(), aux_maps.values(), type_probs.values()))
        seg_results = self.parallel_pipeline(maps)
        return seg_results