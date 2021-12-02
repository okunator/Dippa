import numpy as np
from abc import ABC, abstractmethod 
from pathos.multiprocessing import ThreadPool as Pool
from typing import Tuple, List, Dict
from tqdm import tqdm
from collections import OrderedDict
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters import rank

from src.utils import (
    remove_debris,
    remove_area_debris,
    fill_holes,
    binarize
)

from ._combine_type_inst import combine_inst_type

from ._thresholding import (
    argmax,
    sauvola_thresh,
    niblack_thresh,
    naive_thresh_prob
)


class PostProcessor(ABC):
    def __init__(
            self,
            thresh_method: str="naive",
            thresh: float=0.5
        ) -> None:
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
        if thresh_method not in allowed:
            raise ValueError(f"""
                Illegal thresholding method. Got: {thresh_method}.
                Allowed: {allowed}."""
            )
        
        self.method = thresh_method
        self.thresh = thresh

    @abstractmethod
    def post_proc_pipeline(self):
        raise NotImplementedError
    
    def run_post_processing(
            self,
            out_maps: Dict[str, Dict[str, np.ndarray]]
        ) -> List[Tuple[str, np.ndarray]]:
        """
        Run post proc pipeline in parallel for each set of predictions

        Args:
        -----------
            out_maps: Dict[str, Dict[str, np.ndarray]]:
                The output predictions from the multitask network in a
                dictionary of dictionaries, where the outer keys are
                the names of te out map types e.g. "inst_map","type_map"
                etc. and the inner dictionary keys are filenames and
                and values are the out maps. E.g.
                
                {
                    "inst_map: {
                        "sample1": np.ndarray, "sample2": np.ndarray
                    }
                    "type_map": {
                        "sample1": np.ndarray, "sample2": np.ndarray
                    } 
                } 

        Returns:
        -----------
            List: A list of dicts containing filename and map type keys, 
                  mapped to post-processed seg and aux maps
                  
                  The output maps depend on the outputs of the network.
                  If the network does not output aux, type or sem maps,
                  these are not contained in the result list.
            
            Output example:
            
            [
                ({"fn": "sample1"}, {"aux": aux_map}, {"inst": inst_map}...)
                ({"fn": "sample1"}, {"aux": aux_map}, {"inst": inst_map}...)
            ]
        """
        map_types = list(out_maps.keys())
        fnames = out_maps[map_types[0]].keys()
        
        maps = []
        for name in fnames:
            inp = ({"fn": name}, )
            for mtype in map_types:
                inp += ({mtype: out_maps[mtype][name]}, )
            maps.append(inp)
        
        seg_results = []
        with Pool() as pool:
            for x in tqdm(
                pool.imap_unordered(self.post_proc_pipeline, maps), 
                total=len(maps), desc=f"Post-processing"
            ):
                seg_results.append(x)

        return seg_results

    def _threshold_probs(
        self,
        maps: List[Tuple[Dict[str, np.ndarray]]]
    ) -> Dict[str, np.ndarray]:
        """
        Assemble network output maps in a dictionary for simple access
        and threshold the instance and sem probabilities

        Args:
        ---------
            maps (Tuple[Dict[str, np.ndarray]]):
                List of the post processing input values.
                
        Example of `maps`:
        ----------
        ({"fn": "sample1"}, {"aux_map": np.ndarray}, {"inst_map": np.ndarray})
        
        Returns:
        ---------
            Dict: A dictionary containing all the resulting pred maps
                  from the network plus the filename of the sample.
        """
        
        out_maps = OrderedDict()
        for d in maps:
            for k, map in d.items():
                if k == "inst_map":
                    out_maps[k] = self._threshold(map) # uint32
                elif k == "type_map":
                    out_maps[k] = argmax(map, axis=-1) # uint32
                elif k == "sem_map":
                    out_maps[k] = self._process_sem_map(map) 
                else:
                    out_maps[k] = map
                    
        if "inst_map" not in out_maps.keys():
            try:
                out_maps["inst_map"] = binarize(out_maps["type_map"])
            except KeyError:
                raise KeyError(f"""
                    Neither of 'inst_map' or 'type_map' keys were found
                    in `out_maps`. Got {out_maps.keys()}."""
                )
            except Exception as e:
                print(e)
                
        return out_maps
        
    def _finalize(
        self,
        maps: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finalize the segmentation maps. I.e. remove debris and majority
        decide the cell classes by majority voting

        Args:
        ---------
            maps (List[np.ndarray]):
                A list of the name of the file, soft masks, and hover 
                maps from the network

        Returns:
        ---------
            List[np.ndarray]: All The final seg maps. Shapes: (H, W)
                              dtypes: uint32
        """
        if "type_map" in maps.keys() is not None:
            maps["type_map"] = combine_inst_type(
                maps["inst_map"], maps["type_map"]
            )
        
        # Clean up the result
        maps["inst_map"] = remove_debris(maps["inst_map"], min_size=10)

        return maps


    def _threshold(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Thresholds the probability map. 

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
        labels = remove_area_debris(labels, min_size=10000)
        labels = fill_holes(labels, min_size=10000)

        return labels.astype("u4")
