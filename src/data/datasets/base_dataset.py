import torch
import numpy as np
import zarr
import tables as tb
import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Dict

from src.dl.utils import minmax_normalize_torch
from src.utils import (
    FileHandler,
    get_weight_map, 
    remove_1px_boundary, 
    fix_duplicates,
    binarize
)


class BaseDataset(Dataset, FileHandler):
    def __init__(
            self,
            fname:str,
            transforms: A.Compose,
            normalize_input: bool=False,
            rm_touching_nuc_borders: bool=False,
            edge_weights: bool=False,
            type_branch: bool=True,
            semantic_branch: bool=False,
        ) -> None:
        """
        Base dataset class

        Args:
        ----------
            fname (str): 
                path to the zarr/hdf5 database
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
            edge_weights (bool, default=False):
                If True, each dataset iteration will create weight maps
                for the nuclear edges. This can be used to penalize
                nuclei edges in cross-entropy based loss functions.
            type_branch (bool, default=False):
                If cell type branch is included in the model, this arg
                signals that the cell type annotations are included per
                each dataset iter. Given that these annotations exist in
                db
            semantic_branch (bool, default=False):
                If the model contains a semnatic area branch, this arg 
                signals that the area annotations are included per each 
                dataset iter. Given that these annotations exist in db
        """
        assert transforms is not None, (
            "No augmentations given. Give at least epmty albu.Compose"
        )

        self.fname = fname
        self.transforms = transforms
        self.normalize_input = normalize_input
        self.rm_touching_nuc_borders = rm_touching_nuc_borders
        self.edge_weights = edge_weights
        self.type_branch = type_branch
        self.semantic_branch = semantic_branch

        self.suffix = Path(self.fname).suffix 
        assert self.suffix in (".h5", ".zarr"), (
            "the input data needs to be in either hdf5 or zarr db"
        )

        # Get the numeber of patches in dataset
        if self.suffix == ".zarr":
            z = zarr.open(self.fname, mode="r")
            self.n_items = z.attrs["n_items"]
        elif self.suffix == ".h5":
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
        (either Zarr/hdf5)

        Returns:
        ---------
            Callable. Either h5 or zarr read method 
        """

        read_func = self.read_zarr_patch
        if self.suffix == ".h5":
            read_func = self.read_h5_patch

        return read_func

    def _get_and_preprocess(self, ix: int) -> Dict[str, np.ndarray]:
        """
        Convenience function to get data from the database and perform
        the operations that are performed for any dataset in this repo.

        Operations in order:
        1. read data from hdf5/zarr db
        2. fix duplicated instances due to mirror padding
        3. generate the weight map for the borders of overlapping nuclei
        4. binarize input for the branch predicting foreground vs. 
           background

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
        insts = self.fix_mirror_pad(insts)

        if self.rm_touching_nuc_borders:
            insts = self.remove_overlaps(insts)
    
        # binarize inst branch mask
        binary_map = self.binary(insts)

        # Collect all the data needed to a dict
        data = {
            "image": im,
            "binary_map": binary_map,
            "inst_map": insts
        }

        if self.edge_weights:
            weight_map = self.generate_weight_map(insts)
            data["weight_map"] = weight_map

        if self.type_branch:
            data["type_map"] = types

        if self.semantic_branch:
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
            img = self.normalize(img)

        # gather the augmented data to a dictionary
        res = {
            k: torch.from_numpy(mask) 
            for k, mask in aug_data.items()
            if k != "image"
        }
        res["image"] = img # this is already a torch tensor so add here
        res["filename"] = self.fname

        return res


    def generate_weight_map(self, inst_map: np.ndarray) -> np.ndarray:
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

    def remove_overlaps(self, inst_map: np.ndarray) -> np.ndarray:
        """
        Remove one pixel from the borders of the nuclei

        Args:
        --------
            inst_map (np.ndarray):
                Instance segmentation map. Shape (H, W)

        Returns:
        --------
            np.ndarray: Resulting instance segmentaiton map. 
            Shape (H, W).
        """
        return remove_1px_boundary(inst_map)

    def fix_mirror_pad(self, inst_map: np.ndarray) -> np.ndarray:
        """
        Fix the redundant indices of the nuclear instances after mirror 
        padding

        Args:
        --------
            inst_map (np.ndarray):
                Instance segmentation map. Shape (H, W)

        Returns:
        --------
            np.ndarray: Resulting instance segmentaiton map. 
            Shape (H, W).
        """
        return fix_duplicates(inst_map)
        
    def binary(self, inst_map: np.ndarray) -> np.ndarray:
        """
        binarize the indice in an instance segmentation map

        Args:
        --------
            inst_map (np.ndarray):
                Instance segmentation map. Shape (H, W)

        Returns:
        --------
            np.ndarray: Binary mask. Shape (H, W).
        """
        return binarize(inst_map)

    def normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        min-max normalize input img tensor

        Args:
        --------
            img (torch.Tensor):
                input tensor image. Shape (C, H, W)

        Returns:
        --------
            torch.Tensor: Normalized input image tensor. 
            Shape (C, H, W).
        """
        return minmax_normalize_torch(img)