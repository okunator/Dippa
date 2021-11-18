import torch
from typing import List,Dict, Tuple

from ._base._base_datamodule import BaseDataModule
from src.dl.utils import to_device


class CustomDataModule(BaseDataModule):
    def __init__(
            self,
            train_db_path: str,
            test_db_path: str,
            target_types: List[str],
            dataset_type: str="hover",
            augs: List[str]=["hue_sat", "non_rigid", "blur"],
            normalize: bool=False,
            return_weight_map: bool=False,
            rm_touching_nuc_borders: bool=False,
            batch_size: int=8,
            num_workers: int=8,
            **kwargs
        ) -> None:
        """
        Set up a custom datamodule that can be used for any h5 db with
        the right format.
        
        Args:
        ---------
            train_db_path (str):
                Path to the hdf5/zarr train database
            test_db_path (str):
                Path to the hdf5/zarr test database
            target_types (List[str]):
                A list of the targets that are loaded during dataloading
                process. Allowed values: "inst", "type", "sem".
            dataset_type (str, default="hover"):
                The dataset type. One of: "hover", "dist", "contour",
                "basic", "unet"
            augs (List, default=["hue_sat","non_rigid","blur"])
                List of augs. Allowed augs: "hue_sat", "rigid",
                "non_rigid", "blur", "non_spatial", "normalize"
            normalize (bool, default=False):
                If True, channel-wise min-max normalization is applied 
                to input imgs in the dataloading process
            return_weight_map (bool, default=False):
                If True, a weight map is loaded during dataloading
                process for weighting nuclear borders.
            rm_touching_nuc_borders (bool, default=False):
                If True, the pixels that are touching between distinct
                nuclear objects are removed from the masks.
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading 
                process.
        """
        allowed = ("inst", "type", "sem")
        if (not all(k in allowed for k in target_types)):
            raise ValueError(f"""
                Allowed values for `target_types`: {allowed}. Got:
                {target_types}"""
            )
        
        super().__init__(
            train_db_path,
            test_db_path,
            target_types,
            dataset_type,
            augs,
            normalize,
            return_weight_map,
            rm_touching_nuc_borders,
            batch_size,
            num_workers
        )
                
    @property
    def class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the proportion of pixels of diffenrent classes in the
        train dataset
        """
        weights = self.get_class_weights(self.db_fname_train.as_posix())
        weights_bin = self.get_class_weights(
            self.db_fname_train.as_posix(), binary=True
        )
        return to_device(weights), to_device(weights_bin)

    @property
    def class_dicts(self) -> Tuple[Dict[str, int]]:
        """
        Get the cell type and possible semantic classes of this dataset. 
        These should be saved in the db
        """
        return self.get_class_dicts(self.db_fname_train.as_posix())