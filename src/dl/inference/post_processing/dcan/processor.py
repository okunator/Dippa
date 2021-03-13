import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
from pathos.multiprocessing import ThreadPool as Pool

from ..base_processor import PostProcessor
from .post_proc import post_proc_dcan


class DcanPostProcessor(PostProcessor):
    def __init__(self,
                 thresh_method: str="naive",
                 thresh: float=0.5,
                 **kwargs) -> None:
        """
        Wrapper class for the DCAN post-processing pipeline for networks outputting
        contour maps. 

        https://arxiv.org/abs/1604.02677

         Args:
        ----------
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the insntance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
        """
        super(DcanPostProcessor, self).__init__(thresh_method, thresh)

    def post_proc_pipeline(self, maps: List[np.ndarray]) -> Tuple[np.ndarray]:
        """
        1. Run the dcan post-proc.
        2. Combine type map and instance map

        Args:
        -----------
            maps (List[np.ndarray]):
                A list of the name of the file, soft mask, and contour map from the network
        """
        name = maps[0]
        prob_map = maps[1]
        contour_map = maps[2]
        type_map = maps[3]

        inst_map = post_proc_dcan(prob_map[..., 1], contour_map.squeeze())

        types = None
        combined = None
        if type_map is not None:
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
        ----------
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