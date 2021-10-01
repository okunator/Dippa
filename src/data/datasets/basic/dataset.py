import torch
import albumentations as A
from typing import Dict

from ..base_dataset import BaseDataset 


class BasicDataset(BaseDataset):
    def __init__(
            self,
            fname: str,
            transforms: A.Compose,
            normalize_input: bool=False,
            rm_touching_nuc_borders: bool=False,
            edge_weights: bool=False,
            type_branch: bool=True,
            semantic_branch: bool=False,
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
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
            edge_weights (bool, default=False):
                If True, each dataset iteration will create weight maps
                for the nuclear edges. This can be used to penalize
                nuclei edges in cross-entropy based loss functions.
            type_branch (bool, default=False):
                If cell type branch is included in the model, this arg
                signals that the cell type annotations are included per
                each dataset iter. Given that these annotations exist in
                db
            semantic_branch (bool, default=False):
                If the model contains a semnatic area branch, this arg 
                signals that the area annotations are included per each 
                dataset iter. Given that these annotations exist in db
        """
        super(BasicDataset, self).__init__(
            fname,
            transforms,
            normalize_input,
            rm_touching_nuc_borders,
            edge_weights,
            type_branch,
            semantic_branch
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
        data = self._get_and_preprocess(ix)
        aug_data = self._augment(data)

        return aug_data