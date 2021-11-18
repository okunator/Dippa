import pytorch_lightning as pl
from copy import deepcopy
from pathlib import Path
from typing import List, Optional
from torch.utils.data import DataLoader

from src.utils import FileHandler
from ...datasets.utils import prepare_dataset


class BaseDataModule(pl.LightningDataModule, FileHandler):
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
            num_workers: int=8
        ) -> None:
        """
        Sets up a datamodule for the given h5/zarr databases.
        The databases need to be written with the writers of this repo.

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
        super().__init__()
        
        self.db_fname_train = Path(train_db_path)
        self.db_fname_test = Path(test_db_path)
        
        self.augs = augs
        self.norm = normalize
        self.rm_nuc_borders = rm_touching_nuc_borders
        self.return_weight_map = return_weight_map
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_type = dataset_type
        
        self.target_types = deepcopy(target_types)
                
        if dataset_type in ("hover", "dist", "contour"):
            self.target_types.append("aux")

    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = prepare_dataset(
            fname=self.db_fname_train.as_posix(),
            name=self.dataset_type,
            phase="train",
            augs=self.augs,
            input_size=256,
            target_types=self.target_types,
            normalize_input=self.norm,
            return_weight_map = self.return_weight_map,
            rm_touching_nuc_borders=self.rm_nuc_borders,

        )
        self.validset = prepare_dataset(
            fname=self.db_fname_train.as_posix(),
            name=self.dataset_type,
            phase="valid",
            augs=self.augs,
            input_size=256,
            target_types=self.target_types,
            normalize_input=self.norm,
            return_weight_map = self.return_weight_map,
            rm_touching_nuc_borders=self.rm_nuc_borders,

        )
        self.testset = prepare_dataset(
            fname=self.db_fname_train.as_posix(),
            name=self.dataset_type,
            phase="test",
            augs=self.augs,
            input_size=256,
            target_types=self.target_types,
            normalize_input=self.norm,
            return_weight_map = self.return_weight_map,
            rm_touching_nuc_borders=self.rm_nuc_borders,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validset, 
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset,
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=self.num_workers
        )
