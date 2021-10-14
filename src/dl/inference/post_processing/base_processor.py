import numpy as np
from abc import ABC, abstractmethod 
from pathos.multiprocessing import ThreadPool as Pool
from typing import Tuple, List, Dict
from tqdm import tqdm
from collections import OrderedDict
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters import rank

from .combine_type_inst import combine_inst_semantic
from src.utils import (
    remove_debris, remove_area_debris, fill_holes
)
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

    def _threshold_probs(
        self,
        maps: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Assemble network output maps in a dictionary for simple access
        and threshold the instance and sem probabilities

        Args:
        ---------
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks and auxiliary
                maps

        Returns:
        ---------
            Dict: A dictionary containing all the resulting pred maps
                  from the network plus the filename of the sample.
                  Order of return masks: "aux", "inst", "types", "sem"
        """
        # TODO: this will fail if the order of maps
        keys = ["fn", "inst_probs", "type_probs", "sem_probs", "aux_map"]
        maps = OrderedDict({k: v for k, v in zip(keys, maps)})
        maps["inst_map"] = self.threshold(maps["inst_probs"])

        if maps["type_probs"] is not None:
            maps["type_map"] = np.argmax(maps["type_probs"], axis=-1)

        if maps["sem_probs"] is not None:
            maps["sem_map"] = self._process_sem_map(maps["sem_probs"])

        return maps

    def _finalize_inst_seg(
        self,
        maps: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finalize the instance segmentation by combining the type maps
        and post-processed instance maps into one.

        Args:
        ---------
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks, and hover 
                maps from the network

        Returns:
        ---------
            Tuple: The final inst map and combined inst and type seg map
        """
        combined = None
        if maps["type_map"] is not None:
            combined = combine_inst_semantic(
                maps["inst_map"], maps["type_map"]
            )
        
        # Clean up the result
        inst_map = remove_debris(maps["inst_map"], min_size=10)

        return inst_map, combined

    def _parallel_pipeline(
            self,
            maps: List[Tuple[np.ndarray]]
        ) -> List[Tuple[str, np.ndarray]]:
        """
        Run post proc pipeline in parallel for each set of predictions

        Args:
        -----------
            maps (List[Tuple[np.ndarray]]):
                A list of tuples containing the fname, inst_map, 
                aux_map, and type_map to be post processed.

        Returns:
        -----------
            List: A list of tuples containing filename (str), 
                  post-processed inst, aux, sem, and type map (ndarray).
                  
                  The output maps depend on the outputs of the network.
                  If the network does not output aux, type or sem maps,
                  these are not contained in the result list.
            
            Output example:
            [("filename1", aux_map, inst_map, type_map, sem_map),
             ("filename2", aux_map, inst_map, type_map, sem_map)]
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


    def _smoothen(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Mean filter a probability map of shape (H, W, C)

        Args:
        ----------
            prob_map (np.ndarray, np.float64): 
                probability map from the any branch of the network. 
                Shape (H, W, C).
            
        Returns:
        ----------
            np.ndarray: Smoothed prob map. Shape (H, W, C).
        """
        smoothed = np.zeros_like(prob_map)
        for i in range(prob_map.shape[-1]):
            smoothed[..., i] = rank.mean(
                img_as_ubyte(prob_map[..., i]), selem=disk(25)
            )
        
        return smoothed

    def _process_sem_map(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Post process the semantic probability map from the network. 

        Args:
        ----------
            prob_map (np.ndarray, np.float64): 
                probability map from the sem branch of the network. 
                Shape (H, W, C_sem).
            
        Returns:
        ----------
            np.ndarray: Indice map. Integer valued mask of the semantic
                        areas. Shape (H, W).
        """
        probs = self._smoothen(prob_map)
        labels = np.argmax(probs, axis=-1)
        labels = remove_area_debris(labels)
        labels = fill_holes(labels)

        return labels
