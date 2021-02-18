import tables
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional

from src.img_processing.process_utils import binarize
from src.utils.file_manager import FileHandler
from .pre_proc_utils import get_weight_map, remove_1px_boundary, fix_mirror_padding


class BaseDataset(Dataset, FileHandler):
    def __init__(self,
                 fname: str,
                 transforms: List) -> None:
        """
        Base dataset class

        Args:
            fname (str): 
                path to the pytables database
            transforms (albu.Compose): 
                albumentations.Compose obj (a list of augmentations)
        """
        assert transforms is not None, "No augmentations given. Give at least epmty albu.Compose"
        self.fname = fname
        self.transforms = transforms
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
