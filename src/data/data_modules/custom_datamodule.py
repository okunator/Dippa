import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from torch.utils.data import DataLoader

from src.utils import FileHandler
from src.dl.utils import to_device
from ..datasets.dataset_builder import DatasetBuilder


class CustomDataModule(pl.LightningDataModule, FileHandler):
    def __init__(
        self,
        train_db_path: str,
        test_db_path: str,
        classes: Dict[str, int],
        augmentations: List[str]=["hue_sat", "non_rigid", "blur"],
        normalize: bool=False,
        aux_branch: str="hover",
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
            classes (Dict[str, int]):
                A dictionary of classes in the dataset. E.g. {"bg":0 ..}
            augmentations (List, default=["hue_sat","non_rigid","blur"])
                List of augmentations. Allowed augs: "hue_sat", "rigid",
                "non_rigid", "blur", "non_spatial", "normalize"
            normalize (bool, default=False):
                If True, channel-wise min-max normalization is applied 
                to input imgs in the dataloading process
            aux_branch (str, default="hover"):
                Signals that the dataset needs to prepare an input for 
                an auxiliary branch in the __getitem__ method. One of: 
                "hover", "dist", "contour", None. If None, assumes that
                the network does not contain auxiliary branch and the
                unet style dataset (edge weights) is used as the dataset 
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading 
                process.
        """
        super(CustomDataModule, self).__init__()

        self.db_fname_train = Path(train_db_path)
        self.db_fname_test = Path(test_db_path)
        self.classes = classes
        
        self.augs = augmentations
        self.norm = normalize
        self.aux_branch = aux_branch
        self.batch_size = batch_size
        self.num_workers = num_workers

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

    def setup(self, stage: Optional[str]=None) -> None:
        self.trainset = DatasetBuilder.set_train_dataset(
            decoder_aux_branch=self.aux_branch,
            augmentations=self.augs,
            fname=self.db_fname_train.as_posix(),
            normalize_input=self.norm
        )
        self.validset = DatasetBuilder.set_test_dataset(
            decoder_aux_branch=self.aux_branch,
            fname=self.db_fname_test.as_posix(),
            normalize_input=self.norm
        )
        self.testset = DatasetBuilder.set_test_dataset(
            decoder_aux_branch=self.aux_branch,
            fname=self.db_fname_test.as_posix(),
            normalize_input=self.norm
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


    
