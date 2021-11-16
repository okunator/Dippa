import torch
import albumentations as A
from typing import Dict

from .pre_proc import gen_hv_maps
from .._base._base_dataset import BaseDataset


class HoverDataset(BaseDataset):
    def __init__(
            self,
            fname: str,
            transforms: A.Compose,
            target_types: Dict[str, bool],
            normalize_input: bool=False,
            rm_touching_nuc_borders: bool=False,
        ) -> None:
        """
        Dataset where masks are pre-processed similarly to the Hover-Net
        paper: https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045

        Args:
        ----------
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
        super(HoverDataset, self).__init__(
            fname,
            transforms,
            target_types,
            normalize_input,
            rm_touching_nuc_borders,
        )

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        """
        Read and pre-process all the data and apply augmentations
        Creates auxilliary nuclei horizontal and vertical gradient maps 
        as extra inputs for the network.

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

        # generate hv-maps
        hvmaps = gen_hv_maps(data["inst_map"])
        data["xmap"] = hvmaps["xmap"]
        data["ymap"] = hvmaps["ymap"]
        
        # augment everything
        aug_data = self._augment(data)

        # TODO: Test if this is faster to do here
        # stack the horizontal and vertical gradients, xy-order
        aug_data["aux_map"] = torch.stack(
            [aug_data["xmap"], aug_data["ymap"]],
            dim=0
        )

        # delete redundant data
        del aug_data["xmap"]
        del aug_data["ymap"]

        return aug_data