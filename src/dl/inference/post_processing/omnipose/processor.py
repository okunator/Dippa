import numpy as np
from typing import Dict, Tuple

from .post_proc import post_proc_omnipose
from .._base._base_processor import PostProcessor


class OmniposePostProcessor(PostProcessor):
    def __init__(
            self,
            thresh_method: str="naive",
            thresh: float=0.5,
            **kwargs
        ) -> None:
        """
        Wrapper class to run the OmniPose post processing pipeline for 
        networks outputting instance maps, Optional[type maps], 
        and horizontal & vertical flows.

        OmniPose:
        https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

        Args:
        -----------
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance
                branch. One of: "naive", "argmax", "sauvola", "niblack".
            thresh (float, default = 0.5): 
                threshold prob value. Used if `thresh_method` == "naive"
        """
        super().__init__(thresh_method, thresh)
        self.flows = None

    def post_proc_pipeline(
            self,
            maps: Tuple[Dict[str, np.ndarray]],
            save_flows: bool=False,
        ) -> Dict[str, np.ndarray]:
        """
        Run the Cellpose post-poc pipeline + additional refining

        Args:
        -----------
            maps (Tuple[Dict[str, np.ndarray]]):
                List of the post processing input values.
            save_flows (bool, default=False):
                Save flows to a class attribute
            
                
        Example of `maps`:
        -----------
        ({"fn": "sample1"}, {"aux_map": np.ndarray}, {"inst_map": np.ndarray})
                
        Returns:
        ----------
            Dict[str, np.ndarray]: A dictionary of the out type mapped
                                   to a numpy array
        """
        maps = self._threshold_probs(maps)
        aux_key = self._get_aux_key(list(maps.keys()))
               
        inst_map = post_proc_omnipose(
            flow_map=maps[aux_key],
            inst_map=maps["inst_map"],
            return_flows=save_flows
        )
        
        if isinstance(inst_map, tuple):
            inst_map = inst_map[0]
            self.flows = inst_map[1]
            
        maps["inst_map"] = inst_map
        maps = self._finalize(maps)

        res = {
            key: map for key, map in maps.items() 
            if not any([l in key for l in ("probs", "aux")])
        }

        return res
    