import tables
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from src.utils.file_manager import FileHandler
from ..torch_utils import percentile_normalize_torch

from src.utils.mask_utils import (
    get_weight_map, 
    remove_1px_boundary, 
    fix_mirror_padding,
    binarize
)


class BaseDataset(Dataset, FileHandler):
    def __init__(self,
                 fname: str) -> None:
        """
        Base dataset class

        Args:
            fname (str): 
                path to the pytables database
        """
        self.fname = fname
        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.n_items = self.tables.root.img.shape[0]
        self.tables.close()

    def __len__(self): 
        return self.n_items

    def generate_weight_map(self, inst_map: np.ndarray) -> np.ndarray: 
        wmap = get_weight_map(inst_map)
        wmap += 1 # uniform weight for all classes
        return wmap

    def remove_overlaps(self, inst_map: np.ndarray) -> np.ndarray:
        return remove_1px_boundary(inst_map)

    def fix_mirror_pad(self, inst_map: np.ndarray) -> np.ndarray:
        return fix_mirror_padding(inst_map)
        
    def binary(self, inst_map:np.ndarray) -> np.ndarray:
        return binarize(inst_map)

    def normalize(self, img: torch.Tensor) -> torch.Tensor:
        return percentile_normalize_torch(img)