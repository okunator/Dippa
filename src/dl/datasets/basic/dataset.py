import torch
import numpy as np
from typing import List, Optional, Dict
from src.dl.datasets.base_dataset import BaseDataset 


class BasicDataset(BaseDataset):
    def __init__(self,
                 fname: str,
                 transforms: List,
                 normalize_input: bool=False) -> None:
        """
        Basic dataset without any task-specific pre-processing for labels

        Args:
        -----------
            fname (str): 
                Path to the pytables database
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            normalize_input (bool, default=False):
                apply percentile normalization to inmut images after transforms
        """
        assert transforms is not None, "No augmentations given. Give at least epmty albu.Compose"
        super(BasicDataset, self).__init__(fname)
        self.transforms = transforms
        self.normalize_input = normalize_input

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        1. read data from hdf5 file
        2. fix duplicated instances due to mirror padding
        3. generate the weight map for the borders of overlapping nuclei
        4. binarize input for the branch predicting foreground vs. background
        5. augment
        """
        im_patch, inst_patch, type_patch = self.read_patch(self.fname, index)
        inst_patch = self.fix_mirror_pad(inst_patch)
        weight_map = self.generate_weight_map(self.remove_overlaps(inst_patch))
    
        # binarize inst branch mask
        inst_patch = self.binary(inst_patch)

        # augment (albumentations)
        augmented_data = self.transforms(image=im_patch, masks=[inst_patch, type_patch, weight_map])
        img = augmented_data["image"]
        masks = augmented_data["masks"]

        if self.normalize_input:
            img = self.normalize(img)

        result = {
            "image": img,
            "binary_map": torch.from_numpy(masks[0]),
            "type_map": torch.from_numpy(masks[1]),
            "weight_map": torch.from_numpy(masks[2]),
            "filename": self.fname
        }
        return result