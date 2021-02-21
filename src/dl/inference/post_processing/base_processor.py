from abc import ABC, abstractmethod 
import numpy as np

import src.dl.inference.post_processing as post_proc


class PostProcessor(ABC):
    def __init__(self, thresh_method: str="naive", thresh: float=0.5) -> None:
        """
        Base class for post processors

        Args:
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the insntance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
        """
        assert thresh_method in post_proc.THRESH_LOOKUP.keys(), (
            f"method not one of {list(post_proc.THRESH_LOOKUP.keys())}"
        )
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
        key = post_proc.THRESH_LOOKUP[self.method]
        return post_proc.__dict__[key](**kwargs) 

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
        return post_proc.combine_inst_semantic(inst_map, type_map)

    @abstractmethod
    def post_proc_pipeline(self):
        raise NotImplementedError

    @abstractmethod
    def run_post_processing(self):
        raise NotImplementedError
        


    