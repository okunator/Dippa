import numpy as np
from abc import ABC, abstractmethod 
from pathos.multiprocessing import ThreadPool as Pool
from typing import Tuple, List
from tqdm import tqdm

from .thresholding import argmax, sauvola_thresh, niblack_thresh, naive_thresh_prob
from .combine_type_inst import combine_inst_semantic
from src.utils.mask_utils import remove_debris


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
                Thresholding method for the soft masks from the insntance branch.
                One of ("naive", "argmax", "sauvola", "niblack").
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
        """
        methods = ("naive", "argmax", "sauvola", "niblack")
        assert thresh_method in methods, f"method: {thresh_method} not one of {methods}"
        
        self.method = thresh_method
        self.thresh = thresh

    @abstractmethod
    def post_proc_pipeline(self):
        raise NotImplementedError

    @abstractmethod
    def run_post_processing(self):
            raise NotImplementedError

    def parallel_pipeline(self, maps: List[Tuple[np.ndarray]]) -> List[Tuple[np.ndarray]]:
        """
        Run post proc pipeline in parallel for each set of predictions

        Args:
        -----------
            maps (List[Tuple[np.ndarray]]):
                A list of tuples containing the inst_map, Optional[aux_map], Optional[type_map]
                to be post processed.

        Returns:
        -----------
            A list of tuples containing filename, post-processed inst map and type map
            e.g. ("filename1", inst_map: np.ndarray, type_map: np.ndarray)
        """
        seg_results = []
        with Pool() as pool:
            for x in tqdm(pool.imap_unordered(self.post_proc_pipeline, maps), total=len(maps)):
                seg_results.append(x)

        return seg_results

    def threshold(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Thresholds the probability map from the network. 
        Available methods in .thresholding.py

        Args:
        ----------
            prob_map (np.ndarray, np.float64): 
                probability map from the inst branch of the network. Shape (H, W, 2).
            
        Returns:
        ----------
            Thresholded integer valued mask (np.ndarray)
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

    def combine_inst_type(self, inst_map: np.ndarray, type_map: np.ndarray) -> np.ndarray:
        """
        Combines the nuclei types and instances 

        Args:
        -----------
            inst_map (np.ndarray, np.uint32):
                The instance map. (Output from post-processing). Shape (H, W)
            type_map (np.ndarray, np.uint32):
                The type map. (Probabilities from type cls branch of the network). Shape (H, W, C)

        Returns:
        -----------
            The final combined prediction.
        """
        types = np.argmax(type_map, axis=2)
        return combine_inst_semantic(inst_map, types)

    def clean_up(self, inst_map: np.ndarray, min_size: int=10) -> np.ndarray:
        """
        Remove small objects. Sometimes ndimage and skimage does not work properly.

        Args:
        ---------
            inst_map (np.ndarray):
                The input inst map. Shape (H, W)
            min_size (int):
                min size for image objects (number of pixels)

        Returns:
        ---------
            np.ndarray cleaned up inst map. Same shape as input
        """
        return remove_debris(inst_map, min_size)


        


    