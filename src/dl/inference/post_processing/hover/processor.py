import numpy as np
from typing import Dict, Optional, Tuple, List
from pathos.multiprocessing import ThreadPool as Pool
from tqdm import tqdm

from .post_proc import post_proc_hover
from ..base_processor import PostProcessor


class HoverNetPostProcessor(PostProcessor):
    def __init__(self,
                 thresh_method: str,
                 thresh: float=0.5,
                 **kwargs) -> None:
        """
        Wrapper class to run the HoVer-Net post processing pipeline for networks
        outputting instance maps, Optional[type maps], and horizontal & vertical maps.

        HoVer-Net:
        https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

        Args:
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
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks, and hover maps from the network
        """
        name = maps[0]
        prob_map = maps[1]
        hover_map = maps[2]
        type_map = maps[3]

        inst_map = self.threshold(prob_map)
        inst_map = post_proc_hover(inst_map, hover_map)

        types = None
        combined = None
        if type_map is not None:
            types = np.argmax(type_map, axis=2)
            combined = self.combine_inst_type(inst_map, types)

        return name, inst_map, combined

    def run_post_processing(self,
                            inst_maps: Dict[str, np.ndarray],
                            hover_maps: Dict[str, np.ndarray],
                            type_maps: Dict[str, np.ndarray]):
        """
        Run post processing for all predictions

        Args:
            inst_maps (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, soft instance map) pairs 
            aux_maps (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, hover map) pairs.
                hover_map[..., 0] = horizontal map
                hover_map[..., 1] = vertical map
            type_maps (OrderedDict[str, np.ndarray]):
                Ordered dict of (file name, type map) pairs.
                type maps are in one hot format.
        """
        # Set arguments for threading pool
        maps = list(zip(inst_maps.keys(), inst_maps.values(), hover_maps.values(), type_maps.values()))

        # Run post processing
        seg_results = []
        with Pool() as pool:
            for x in tqdm(pool.imap_unordered(self.post_proc_pipeline, maps), total=len(maps)):
                seg_results.append(x)

        return seg_results