import numpy as np
import tables as tb
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Callable

from src.utils import FileHandler
from src.dl.utils import minmax_normalize_torch, binarize
from ._transforms._composition import ToTensorV3


__all__ = ["SegDataset"]


class SegDataset(Dataset, FileHandler):
    def __init__(
            self,
            fname: str,
            img_transforms: Callable,
            inst_transforms: Callable,
            seg_targets: List[str],
            normalize_input: bool=False,
        ) -> None:
        """
        Base dataset class

        Args:
        ----------
            fname (str): 
                path to the hdf5 database
            img_transforms (albu.Compose): 
                Albumentations.Compose obj (a list of transformations).
                All the transformations that are applied to the input
                images and corresponding masks
            inst_transforms (ApplyEach):
                ApplyEach obj. (a list of augmentations). All the
                transformations that are applied to only the instance
                labelled masks.
            seg_targets (List[str]):
                A list of target maps needed for the loss computation.
                All the segmentation targets that are not in this list
                are deleted to save memory
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
        """
        self.suffix = Path(fname).suffix 
        if not self.suffix in (".h5"):
            raise ValueError(f"""
                the input data needs to be in hdf5 database.
                Got: {self.suffix}. Allowed: {(".h5")}"""
            )
            
        self.fname = fname
        self.img_transforms = img_transforms
        self.inst_transforms = inst_transforms
        self.normalize_input = normalize_input
        self.seg_targets = seg_targets
        self.to_tensor = ToTensorV3()
 
        # Get the numeber of patches in dataset
        if self.suffix == ".h5":
            with tb.open_file(self.fname) as h5:
                self.n_items = h5.root._v_attrs["n_items"]

        # Get the dataset stats
        self.stats = self.get_dataset_stats(self.fname)

    def __len__(self): 
        return self.n_items

    def __getitem__(self, ix: int) -> Dict[str, np.ndarray]:
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
                  (np.ndarray) mapped to a key
        """
        im, insts, types, areas = self.read_h5_patch(self.fname, ix)
                
        data = {"image": im}
        data["masks"] = [m for m in (insts, types, areas) if m is not None]
        
        aug = self.img_transforms(**data)
        aux = self.inst_transforms(image=aug["image"], inst_map=aug["masks"][0])
        
        data = self.to_tensor(image=aug["image"], masks=aug["masks"], aux=aux)
        
        if self.normalize_input:
            data["image"] = minmax_normalize_torch(data["image"])
        
        out = {"image": data["image"]}
        for m, n in zip(data["masks"], ("inst", "type", "sem")):
            out[f"{n}_map"] = m
        
        for n, aux_map in aux.items():
            out[f"{n}_map"] = aux_map

        # remove redundant targets
        if "inst" not in self.seg_targets:
            del out["inst_map"]
        else:
            out["inst_map"] = binarize(out["inst_map"])

        return out
    