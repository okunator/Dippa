import torch
import numpy as np
from typing import List, Optional, Dict
from src.dl.datasets.base_dataset import BaseDataset 


class MicroDataset(BaseDataset):
    def __init__(self,
                 fname: str,
                 transforms: List) -> None:
        """
        Dataset where masks are pre-processed similarly to the Micro-Net paper

        Args:
            fname (str): path to the pytables database
            transforms (albu.Compose): albumentations.Compose obj (a list of augmentations)
        """

        super(MicroDataset, self).__init__(fname, transforms)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        1. read data from hdf5 file
        2. fix duplicated instances due to mirror padding
        3. remove overlaps in occluded nuclei and generate the weight map for the borders of overlapping nuclei
        4. create border map as in 
        5. binarize input for the branch predicting foreground vs. background
        6. augment
        """
        im_patch, inst_patch, type_patch = self.read_hdf5_patch(self.fname, index)
        inst_patch = self.fix_mirror_pad(inst_patch)
        weight_map = self.generate_weight_map(self.remove_overlaps(inst_patch))

        # generate boundary-maps
        # TODO: generate boundary maps
        
        # binarize inst branch input
        inst_patch = self.binary(inst_patch)

        # augment
        augmented_data = self.transforms(image=im_patch, masks=[inst_patch, type_patch, weight_map])
        img = augmented_data["image"]
        masks = augmented_data["masks"]

        result = {
            "image": img,
            "binary_map": masks[0],
            "type_map": masks[1],
            "weight_map": masks[2],
            "xmap": masks[3],
            "ymap": masks[4],
            "filename": self.fname
        }
        return result