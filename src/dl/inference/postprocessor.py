import torch
import torch.nn as nn
import numpy as np
import src.img_processing.post_processing as post_proc


class PostProcessor:
    """
    Wrapper class for the methods used in post-processing. I.e.
    1. thresholding
    2. post-processing
    3. combining instance maps and type maps

    Should be used for full sized images and not small patches.
    """

    @staticmethod
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

    @staticmethod
    def post_process(self, 
                     inst_map: np.ndarray,
                     prob_map: np.ndarray,
                     aux_map: np.ndarray = None,
                     method1: str = "regular", 
                     method2: str = "default") -> np.ndarray:
        """
        Apply the post-process pipeline. Uses the appropriate method depending on the network architecture.
        For example, if aux branch regresses horizontal and vertical maps, then "cellpose" and "hover" 
        methods are considered. If there is no auxiliary branch in the network, then the "regular" one is used.
        The method architecture is specified in the config.py file. Methods are found in post_processing/...

        Args:
            inst_map (np.ndarray, np.int32):
                The instance map. (Output from thresholding). Shape (H, W)
            prob_map (np.ndarray, np.float64):
                The probability map from the network, (Before thresholding). Shape (H, W)
            aux_map (np.ndarray, np.float64): 
                Output from the auxiliary regression branch of the network. Shape (H, W, C)
                For "hover" and "cellpose", C = 2.
            method1 (str, "regular"):
                The post processing method. One of ["hover", "micro", "cellpose", "regular"] 
            method2 (str, "default")
                One of ["default", "experimental"]. Use default.

        Returns:
            The post-processed intance map (np.ndarray)
        """
        kwargs = {}
        kwargs.setdefault("inst_map", inst_map)
        kwargs.setdefault("prob_map", prob_map)
        kwargs.setdefault("aux_map", aux_map)
        key = post_proc.POST_PROC_LOOKUP[method1][method2]
        return post_proc.__dict__[key](**kwargs)

    @staticmethod
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
        


    