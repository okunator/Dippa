import numpy as np
from typing import Dict, Tuple, List

from .._base._base_processor import PostProcessor
from .post_proc import post_proc_dran


class DRANPostProcessor(PostProcessor):
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
                branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold prob value. Used if `thresh_method` == "naive"
        """
        super().__init__(thresh_method, thresh)

    def post_proc_pipeline(
            self,
            maps: List[np.ndarray]
        ) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Run the DRAN post-processing pipeline + additional refining
        
        Args:
        -----------
            maps (Tuple[Dict[str, np.ndarray]]):
                List of the post processing input values.            
                
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
        maps["inst_map"] = post_proc_dran(
            maps["inst_map"], maps[aux_key].squeeze()
        )
        maps = self._finalize(maps)

        res = {
            key: map for key, map in maps.items() 
            if not any([l in key for l in ("probs", "aux")])
        }

        return res
