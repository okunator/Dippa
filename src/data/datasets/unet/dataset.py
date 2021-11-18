import torch
import albumentations as A
from typing import List, Dict

from .._base._base_dataset import BaseDataset


class UnetDataset(BaseDataset):
    def __init__(
            self,
            fname: str,
            transforms: A.Compose,
            target_types: List[str],
            normalize_input: bool=False,
            **kwargs
    ) -> None:
        """
        Dataset where masks are pre-processed similarly to the U-net 
        paper: https://arxiv.org/abs/1505.04597.

        Args:
        -----------
            fname (str): 
                Path to the pytables database
            target_types (List[str]):
                A list of the targets that are loaded during dataloading
                process. Allowed values: "inst", "type", "sem".
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
        """
        super(UnetDataset, self).__init__(
            fname,
            transforms,
            target_types,
            normalize_input,
            rm_touching_nuc_borders=True,
            return_weight_map=True
        )

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        """
        Read and pre-process all the data and apply augmentations.
        Generates weight maps for loss weighting and remove overlapping
        nuclei borders.

        Args:
        --------
            ix (int):
                index of the iterable dataset

        Returns:
        --------
            Dict: A dictionary containing all the augmented data patches
                  (torch.Tensor) and the filename of the patches
        """
        data = self._read_and_preprocess(ix)
        aug_data = self._augment(data)

        return aug_data


