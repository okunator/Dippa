import numpy as np
from abc import ABC, abstractmethod 
from pathos.multiprocessing import ThreadPool as Pool
from typing import Tuple, List
from tqdm import tqdm

from src.utils import remove_debris
from .combine_type_inst import combine_inst_semantic
from .thresholding import (
    argmax, sauvola_thresh, niblack_thresh, naive_thresh_prob
)


THRESH_LOOKUP = {
    "argmax":"argmax",
    "sauvola":"sauvola_thresh",
    "niblack":"niblack_thresh",
    "naive":"naive_thresh_prob"
}


class PostProcessor(ABC):
    def __init__(self, thresh_method: str="naive", thresh: float=0.5) -> None:
        """
        Base class for post processors

        Args:
        ----------
            thresh_method (str, default="naive"):
                Thresholding method for soft masks from the insntance 
                branch. One of: "naive", "argmax", "sauvola", "niblack".
            thresh (float, default = 0.5):
                Threshold prob value. Used if `thres_method` == "naive"
        """
        allowed = ("naive", "argmax", "sauvola", "niblack")
        assert thresh_method in allowed, (
            f"method: {thresh_method} not one of {allowed}"
        )
        
        self.method = thresh_method
        self.thresh = thresh

    @abstractmethod
    def post_proc_pipeline(self):
        raise NotImplementedError

    @abstractmethod
    def run_post_processing(self):
            raise NotImplementedError

    def parallel_pipeline(
            self,
            maps: List[Tuple[np.ndarray]]
        ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Run post proc pipeline in parallel for each set of predictions

        Args:
        -----------
            maps (List[Tuple[np.ndarray]]):
                A list of tuples containing the fname, inst_map, 
                aux_map, and type_map to be post processed.

        Returns:
        -----------
            List: a list of tuples containing filename, post-processed 
            inst map and type map.
            
            Example:
            [("filename1", inst_map: np.ndarray, type_map: np.ndarray),
             ("filename2", inst_map: np.ndarray, type_map: np.ndarray)]
        """
        seg_results = []
        with Pool() as pool:
            for x in tqdm(
                pool.imap_unordered(self.post_proc_pipeline, maps), 
                total=len(maps), desc=f"Post-processing"
            ):
                seg_results.append(x)

        return seg_results

    def threshold(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Thresholds the probability map from the network. 

        Args:
        ----------
            prob_map (np.ndarray, np.float64): 
                probability map from the inst branch of the network. 
                Shape (H, W, 2).
            
        Returns:
        ----------
            np.ndarray: Thresholded integer valued mask. Shape (H, W) 
        """
        kwargs = {}
        kwargs["prob_map"] = prob_map
        kwargs["thresh"] = self.thresh

        if self.method == "argmax":
            result = argmax(**kwargs)
        elif self.method == "naive":
            result = naive_thresh_prob(**kwargs)
        elif self.method == "sauvola":
            result = sauvola_thresh(**kwargs)
        elif self.method == "niblack":
            result = niblack_thresh(**kwargs)

        return result

    def combine_inst_type(
            self,
            inst_map: np.ndarray,
            type_map: np.ndarray
        ) -> np.ndarray:
        """
        Combines the nuclei types and instances 

        Args:
        -----------
            inst_map (np.ndarray):
                The post-processed instance map. Shape (H, W)
            type_map (np.ndarray):
                The soft type map. Shape (H, W, C)

        Returns:
        -----------
            np.ndarray: The final combined inst_map + type_map 
            prediction (instance segmentation). Shape (H, W) 
        """
        types = np.argmax(type_map, axis=2)
        return combine_inst_semantic(inst_map, types)

    def clean_up(self, inst_map: np.ndarray, min_size: int=10) -> np.ndarray:
        """
        Remove small objects. Sometimes the ndimage and skimage methods 
        do not work as they should and fail to remove small objs...

        Args:
        ---------
            inst_map (np.ndarray):
                The input inst map. Shape (H, W)
            min_size (int):
                min size for image objects (number of pixels)

        Returns:
        ---------
            np.ndarray: cleaned up inst map. Shape (H, W)
        """
        return remove_debris(inst_map, min_size)
