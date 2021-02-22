from abc import ABC, abstractmethod 
import numpy as np

from .thresholding import argmax, sauvola_thresh, niblack_thresh, naive_thresh_prob
from .combine_type_inst import combine_inst_semantic

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

    def threshold(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Thresholds the probability map from the network. 
        Available methods in .thresholding.py

        Args:
            prob_map (np.ndarray, np.float64): 
                probability map from the inst branch of the network. Shape (H, W, 2).
            
        Returns:
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
            inst_map (np.ndarray, np.uint32):
                The instance map. (Output from post-processing). Shape (H, W)
            type_map (np.ndarray, np.uint32):
                The type map. (Argmaxed output from type cls branch of the network). Shape (H, W)

        Returns:
            The final combined prediction.
        """
        return combine_inst_semantic(inst_map, type_map)

    @abstractmethod
    def post_proc_pipeline(self):
        raise NotImplementedError

    @abstractmethod
    def run_post_processing(self):
        raise NotImplementedError
        


    