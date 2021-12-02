import torch
import numpy as np
import tables as tb
import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, List, Dict

from src.utils import (
    FileHandler,
    get_weight_map, 
    remove_1px_boundary,
    binarize
)

from src.dl.utils import minmax_normalize_torch


class BaseDataset(Dataset, FileHandler):
    def __init__(
            self,
            fname: str,
            transforms: A.Compose,
            target_types: List[str],
            normalize_input: bool=False,
            return_weight_map: bool=False,
            rm_touching_nuc_borders: bool=False,
        ) -> None:
        """
        Base dataset class

        Args:
        ----------
            fname (str): 
                path to the hdf5 database
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            target_types (List[str]):
                A list of the targets that are loaded during dataloading
                process. Allowed values: "inst", "type", "sem".
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
            return_weight_map (bool, default=False):
                Include a nuclear border weight map in the dataloading
                process
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
        """
        self.suffix = Path(fname).suffix 
        if not self.suffix in (".h5"):
            raise ValueError(f"""
                the input data needs to be in hdf5 database.
                Got: {self.suffix}. Allowed: {(".h5")}"""
            )
            
        self.fname = fname
        self.transforms = transforms
        self.normalize_input = normalize_input
        self.return_weight_map = return_weight_map
        self.rm_touching_nuc_borders = rm_touching_nuc_borders
        self.target_types = target_types
 
        # Get the numeber of patches in dataset
        if self.suffix == ".h5":
            with tb.open_file(self.fname) as h5:
                self.n_items = h5.root._v_attrs["n_items"]

        # Get the dataset stats
        self.stats = self.get_dataset_stats(self.fname)

    def __len__(self): 
        return self.n_items

    @property
    def read_patch(self) -> Callable:
        """
        Property function which determines which db type is being used
        (hdf5)

        Returns:
        ---------
            Callable. Either h5 or read method 
        """
        if self.suffix == ".h5":
            read_func = self.read_h5_patch

        return read_func

    def _read_and_preprocess(self, ix: int) -> Dict[str, np.ndarray]:
        """
        Convenience function to get data from the database and perform
        the operations that are performed for any dataset in this repo.

        Args:
        --------
            ix (int):
                index of the iterable dataset

        Returns:
        --------
            Dict: A dictionary containing all the augmented data patches
                  (np.ndarray) and the filename of the patches
        """
        im, insts, types, areas = self.read_patch(self.fname, ix)

        data = {"image": im}
        
        if self.return_weight_map:
            weight_map = self._generate_weight_map(insts)
            data["weight_map"] = weight_map
            
        if self.rm_touching_nuc_borders:
            insts = remove_1px_boundary(insts)
            
        data["inst_map"] = insts
        
        if "inst" in self.target_types:
            data["binary_map"] = binarize(insts)

        if "type" in self.target_types:
            data["type_map"] = types

        if "sem" in self.target_types:
            data["sem_map"] = areas

        return data

    def _augment(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Augment all the patch-data that is provided in a dictionary.
        Also, normalizes the img patch if normalize_input = True

        Args:
        --------
            data (Dict[str, np.ndarray]):
                A dictionary of (key, np.ndarray)-pairs

        Returns:
        --------
            Dict: A dictionary of (key, np.ndarray)-pairs that are
                  augmented
        """
        # Del redundant inst map. Not needed after aux maps are created
        del data["inst_map"]

        # augment all data
        aug_data = self.transforms(**data)
        
        # Normalize input img
        img = aug_data["image"]
        if self.normalize_input:
            img = minmax_normalize_torch(img)

        # gather the augmented data to a dictionary
        res = {
            k: torch.from_numpy(mask).long()
            for k, mask in aug_data.items()
            if k not in ("image")
        }
        
        res["image"] = img.float()
        res["filename"] = self.fname

        return res

    def _generate_weight_map(self, inst_map: np.ndarray) -> np.ndarray:
        """
        Generate a weight map for the nuclei edges

        Args:
        --------
            inst_map (np.ndarray):
                Instance segmentation map. Shape (H, W)

        Returns:
        --------
            np.ndarray: The nuclei border weight map. Shape (H, W) 
        """
        wmap = get_weight_map(inst_map)
        wmap += 1 # uniform weight for all classes
        return wmap