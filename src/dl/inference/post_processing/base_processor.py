import numpy as np

import src.dl.inference.post_processing as post_proc


class PostProcessor:
    def threshold(self, 
                  prob_map: np.ndarray, 
                  method: str = "argmax", 
                  thresh: float = 0.5) -> np.ndarray:
        """
        Thresholds the probability map from the network. Available methods in post_processing/thresholding.py

        Args:
            prob_map (np.ndarray, np.float64): 
                probability map from the sigmoid/softmax layer of the network. Shape (H, W)
            method (str, default = "argmax"): 
                Thresholding method for the probability map. One of ["argmax", "naive", "sauvola", "niblack"].
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"
        
        Returns:
            Thresholded integer valued mask (np.ndarray)
        """
        key = post_proc.THRESH_LOOKUP[method]
        return post_proc.__dict__[key](prob_map, thresh) 

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
        


    