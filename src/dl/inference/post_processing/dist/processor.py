import numpy as np
from typing import Dict, Optional, Tuple, List
from pathos.multiprocessing import ThreadPool as Pool
from tqdm import tqdm

from .post_proc import post_proc_dist
from ..base_processor import PostProcessor


class DistPostProcessor(PostProcessor):
    def __init__(self,
                 thresh_method: str="naive",
                 thresh: float=0.5,
                 **kwargs) -> None:
        """
        Wrapper class to run the DRFNS post processing pipeline for networks
        outputting instance maps, Optional[type maps], and distance transforms

        DRFNS:
        https://ieeexplore.ieee.org/document/8438559

        Args:
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the insntance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
        """
        super(DistPostProcessor, self).__init__(thresh_method, thresh)

    def post_proc_pipeline(self, maps: List[np.ndarray]) -> Tuple[np.ndarray]:
        """
        1. Threshold
        2. Post process instance map
        3. Combine type map and instance map

        Args:
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks, and hover maps from the network
        """
        name = maps[0]
        prob_map = maps[1]
        dist_map = maps[2]
        type_map = maps[3]

        inst_map = self.threshold(prob_map)
        inst_map = post_proc_dist(dist_map, inst_map)

        types = None
        combined = None
        if type_map is not None:
            types = np.argmax(type_map, axis=2)
            combined = self.combine_inst_type(inst_map, types)
        
        # Clean up the result
        inst_map = self.clean_up(inst_map)

        return name, inst_map, combined

    def run_post_processing(self,
                            inst_maps: Dict[str, np.ndarray],
                            dist_map: Dict[str, np.ndarray],
                            type_maps: Dict[str, np.ndarray]):
        """
        Run post processing for all predictions

        Args:
            inst_maps (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, soft instance map) pairs
                inst_map shapes are (H, W, 2) 
            dist_map (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, dist map) pairs.
                The regressed distance trasnform from auxiliary branch. Shape (H, W, 1)
            type_maps (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, type map) pairs.
                type maps are in one hot format (H, W, n_classes).
        """
        # Set arguments for threading pool
        maps = list(zip(inst_maps.keys(), inst_maps.values(), dist_map.values(), type_maps.values()))

        # Run post processing
        seg_results = []
        with Pool() as pool:
            for x in tqdm(pool.imap_unordered(self.post_proc_pipeline, maps), total=len(maps)):
                seg_results.append(x)

        return seg_results