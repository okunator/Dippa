import torch
import numpy as np
from typing import List, Dict
from src.dl.datasets.base_dataset import BaseDataset 

from .pre_proc import contours


class ContourDataset(BaseDataset):
    def __init__(self,
                 fname: str,
                 transforms: List,
                 norm: bool=False) -> None:
        """
        Dataset where masks are pre-processed similarly to the Micro-Net paper

        Args:
        ----------
            fname (str): 
                Path to the pytables database
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            norm (bool, default=False):
                apply percentile normalization to inmut images after transforms
        """
        assert transforms is not None, "No augmentations given. Give at least epmty albu.Compose"
        super(ContourDataset, self).__init__(fname)
        self.transforms = transforms
        self.norm = norm

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        1. read data from hdf5 file
        2. fix duplicated instances due to mirror padding
        3. remove overlaps in occluded nuclei and generate the weight map for the borders of overlapping nuclei
        4. Compute contours 
        5. binarize input for the branch predicting foreground vs. background
        6. augment
        """
        im_patch, inst_patch, type_patch = self.read_patch(self.fname, index)
        inst_patch = self.fix_mirror_pad(inst_patch)
        inst_patch = self.remove_overlaps(inst_patch)
        weight_map = self.generate_weight_map(inst_patch)

        # generate boundary-maps
        contour = contours(inst_patch, thickness=2)
        
        # binarize inst branch input
        inst_patch = self.binary(inst_patch)

        augmented_data = self.transforms(image=im_patch, masks=[inst_patch, type_patch, weight_map, contour])
        img = augmented_data["image"]
        masks = augmented_data["masks"]

        if self.norm:
            img = self.normalize(img)

        result = {
            "image": img,
            "binary_map": torch.from_numpy(masks[0]),
            "type_map": torch.from_numpy(masks[1]),
            "weight_map": torch.from_numpy(masks[2]),
            "contour": torch.from_numpy(masks[3]),
            "filename": self.fname
        }
        return result