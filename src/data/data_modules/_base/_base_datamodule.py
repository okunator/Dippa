import pytorch_lightning as pl
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
            seg_targets: List[str],
            img_transforms: List[str],
            inst_transforms: List[str],
            normalize: bool=False,
            return_weight_map: bool=False,
            input_size: int=256,
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
            seg_targets (List[str]):
                A list of target maps needed for the loss computation.
                All the segmentation targets that are not in this list
                are deleted to save memory
            img_transforms (albu.Compose): 
                Albumentations.Compose obj (a list of transformations).
                All the transformations that are applied to the input
                images and corresponding masks
            inst_transforms (ApplyEach):
                ApplyEach obj. (a list of augmentations). All the
                transformations that are applied to only the instance
                labelled masks.
            normalize (bool, default=False):
                If True, channel-wise min-max normalization is applied 
                to input imgs in the dataloading process
            return_weight_map (bool, default=False):
                If True, a weight map is loaded during dataloading
                process for weighting nuclear borders.
            input_size (int):
                Size of the height and width of the input images
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading 
                process.
        """
        super().__init__()
        
        self.db_fname_train = Path(train_db_path)
        self.db_fname_test = Path(test_db_path)
        
        self.img_transforms = img_transforms
        self.inst_transforms = inst_transforms
        self.seg_targets = seg_targets
        self.input_size = input_size
        self.norm = normalize
        self.return_weight_map = return_weight_map
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None) -> None:
        self.trainset = prepare_dataset(
            fname=self.db_fname_train,
            phase="train",
            seg_targets=self.seg_targets,
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            input_size=self.input_size,
            normalize_input=self.norm,
            return_weight_map=self.return_weight_map,

        )
        self.validset = prepare_dataset(
            fname=self.db_fname_test,
            phase="valid",
            seg_targets=self.seg_targets,
            img_transforms=[],
            inst_transforms=self.inst_transforms,
            input_size=self.input_size,
            normalize_input=self.norm,
            return_weight_map=self.return_weight_map,
        )
        self.testset = prepare_dataset(
            fname=self.db_fname_test,
            phase="valid",
            seg_targets=self.seg_targets,
            img_transforms=[],
            inst_transforms=self.inst_transforms,
            input_size=self.input_size,
            normalize_input=self.norm,
            return_weight_map=self.return_weight_map,
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
