import torch
import numpy as np
import zarr
import tables as tb
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from src.utils.file_manager import FileHandler
from ..torch_img_utils import minmax_normalize_torch

from src.utils import (
    get_weight_map, 
    remove_1px_boundary, 
    fix_duplicates,
    binarize
)


class BaseDataset(Dataset, FileHandler):
    def __init__(self, fname:str) -> None:
        """
        Base dataset class

        Args:
        ----------
            fname (str): 
                path to the zarr/hdf5 database
        """
        self.fname = fname
        self.suffix = Path(self.fname).suffix 
        assert self.suffix in (".h5", ".zarr"), "the input data needs to be in either hdf5 or zarr db"

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
    def read_patch(self) -> np.ndarray:
        """
        Read an image patch from h5 or zarr db

        Returns:
        ---------
            np.ndarray. Shape (H, W, C)
        """
        read_func = self.read_h5_patch if self.suffix == ".h5" else self.read_zarr_patch
        return read_func

    def generate_weight_map(self, inst_map: np.ndarray) -> np.ndarray:
        """
        Generate a weight map for the nuclei edges

        Args:
        --------
            inst_map (np.ndarray):
                Instance segmentation map. Shape (H, W)

        Returns:
        --------
            np.ndarray. The nuclei border weight map. Shape (H, W) 
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
            np.ndarray. Resulting instance segmentaiton map. Shape (H, W) 
        """
        return remove_1px_boundary(inst_map)

    def fix_mirror_pad(self, inst_map: np.ndarray) -> np.ndarray:
        """
        Fix the redundant indices of the nuclear instances after mirror padding

        Args:
        --------
            inst_map (np.ndarray):
                Instance segmentation map. Shape (H, W)

        Returns:
        --------
            np.ndarray. Resulting instance segmentaiton map. Shape (H, W) 
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
            np.ndarray. Binary mask. Shape (H, W) 
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
            torch.Tensor. Normalized input image tensor. Shape (C, H, W)
        """
        return minmax_normalize_torch(img)