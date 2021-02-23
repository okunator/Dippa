import torch
import numpy as np
from typing import List, Optional, Dict

from .pre_proc import gen_dist_maps
from ..base_dataset import BaseDataset 


class DistDataset(BaseDataset):
    def __init__(self,
                 fname: str,
                 transforms: List) -> None:
        """
        Dataset where masks are processed with distance transform
        for regression

        Args:
            fname (str): 
                path to the pytables database
            transforms (albu.Compose): 
                albumentations.Compose obj (a list of augmentations)
        """

        super(DistDataset, self).__init__(fname, transforms)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        1. read data from hdf5 file
        2. fix duplicated instances due to mirror padding
        3. remove overlaps in occluded nuclei and generate the weight map for the borders of overlapping nuclei
        4. create distance maps
        5. binarize input for the branch predicting foreground vs. background
        6. augment
        """
        im_patch, inst_patch, type_patch = self.read_hdf5_patch(self.fname, index)
        inst_patch = self.fix_mirror_pad(inst_patch)
        weight_map = self.generate_weight_map(self.remove_overlaps(inst_patch))

        # generate dist-maps
        distmap = gen_dist_maps(inst_patch)

        # binarize inst branch input
        inst_patch = self.binary(inst_patch)

        # augment
        augmented_data = self.transforms(image=im_patch, masks=[inst_patch, type_patch, weight_map, distmap])
        img = augmented_data["image"]
        masks = augmented_data["masks"]

        result = {
            "image": img,
            "binary_map": masks[0],
            "type_map": masks[1],
            "weight_map": masks[2],
            "dist_map": masks[3],
            "filename": self.fname
        }
        return result