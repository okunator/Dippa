import torch
import albumentations as A
from typing import List, Dict

from ..base_dataset import BaseDataset 


class UnetDataset(BaseDataset):
    def __init__(
            self,
            fname: str,
            transforms: A.Compose,
            normalize_input: bool=False,
            type_branch: bool=True,
            semantic_branch: bool=False,
            **kwargs
    ) -> None:
        """
        Dataset where masks are pre-processed similarly to the U-net 
        paper: https://arxiv.org/abs/1505.04597.

        Overrides `rm_overalps` and `edge_weights` args in the 
        `experiment.yml`-file. Both of these args are set to `True`.

        Args:
        -----------
            fname (str): 
                Path to the pytables database
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
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
        super(UnetDataset, self).__init__(
            fname,
            transforms,
            normalize_input,
            type_branch,
            semantic_branch,
            edge_weights=True,
            rm_touching_nuc_borders=True,
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
        data = self._get_and_preprocess(ix)
        aug_data = self._augment(data)

        return aug_data


