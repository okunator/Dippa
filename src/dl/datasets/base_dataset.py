import tables
import torch
import numpy as np
import src.img_processing.pre_processing as preproc

from torch.utils.data import Dataset
from typing import List, Dict, Optional
from src.utils.file_manager import FileHandler
from src.img_processing.process_utils import binarize


class BaseDataset(Dataset, FileHandler):
    def __init__(self,
                 fname: str,
                 transforms: List) -> None:
        """
        Base dataset class

        Args:
            fname (str): path to the pytables database
            transforms (albu.Compose): albumentations.Compose obj (a list of augmentations)
        """
        
        self.fname = fname
        self.transforms = transforms
        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.n_items = self.tables.root.img.shape[0]
        self.tables.close()

    def __len__(self): 
        return self.n_items

    def generate_weight_map(self, inst_map: np.ndarray) -> np.ndarray: 
        wmap = preproc.get_weight_map(inst_map)
        wmap += 1 # uniform weight for all classes
        return wmap

    def remove_overlaps(self, inst_map: np.ndarray) -> np.ndarray:
        return preproc.remove_1px_boundary(inst_map)

    def fix_mirror_pad(self, inst_map: np.ndarray) -> np.ndarray:
        return preproc.fix_mirror_padding(inst_map)
        
    def binary(self, inst_map:np.ndarray) -> np.ndarray:
        return binarize(inst_map)

    def generate_hv_maps(self, inst_map: np.ndarray) -> np.ndarray:
        return preproc.gen_hv_maps(inst_map)

