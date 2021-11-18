import torch
import albumentations as A
from typing import Dict, List

from .._base._base_dataset import BaseDataset
from .pre_proc import contours


class ContourDataset(BaseDataset):
    def __init__(
            self,
            fname: str,
            transforms: A.Compose,
            target_types: List[str],
            normalize_input: bool=False,
            return_weight_map: bool=False,
            rm_touching_nuc_borders: bool=False,
        ) -> None:
        """
        Dataset where masks are pre-processed similarly to the DCAN 
        paper: https://arxiv.org/abs/1604.02677

        Args:
        ----------
            fname (str): 
                Path to the pytables database
            transforms (albu.Compose): 
                Albumentations.Compose obj (a list of augmentations)
            target_types (List[str]):
                A list of the targets that are loaded during dataloading
                process. Allowed values: "inst", "type", "sem".
            normalize_input (bool, default=False):
                apply minmax normalization to input images after 
                transforms
            return_weight_map (bool, default=False):
                Include a nuclear border weight map in the dataloading
                process
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
        """
        super(ContourDataset, self).__init__(
            fname,
            transforms,
            target_types,
            normalize_input,
            return_weight_map,
            rm_touching_nuc_borders,
        )

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        """
        Read and pre-process all the data and apply augmentations
        Creates auxilliary nuclei contour maps as extra inputs for the
        network.

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
        
        # generate boundary-maps
        data["aux_map"] = contours(
            data["inst_map"],
            thickness=2
        )

        # augment everything
        aug_data = self._augment(data)

        # TODO: Test if this is faster to do here
        # create an extra dim so that loss computing works
        aug_data["aux_map"] = aug_data["aux_map"].unsqueeze(dim=0)

        return aug_data