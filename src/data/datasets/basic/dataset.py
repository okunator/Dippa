import torch
import albumentations as A
from typing import Dict

from .._base._base_dataset import BaseDataset


class BasicDataset(BaseDataset):
    def __init__(
            self,
            fname: str,
            transforms: A.Compose,
            target_types: Dict[str, bool],
            normalize_input: bool=False,
            rm_touching_nuc_borders: bool=False,
        ) -> None:
        """
        Basic dataset without any task-specific pre-processing for 
        labels

        Args:
        -----------
            fname (str): 
                Path to the pytables database
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            target_types (Dict[str, bool]):
                A dictionary mapping target types to a boolean value.
                Allowed keys: "inst", "type, "sem", "wmap".
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
        """
        super().__init__(
            fname,
            transforms,
            target_types,
            normalize_input,
            rm_touching_nuc_borders,
        )

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        """
        Read and pre-process all the data and apply augmentations

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