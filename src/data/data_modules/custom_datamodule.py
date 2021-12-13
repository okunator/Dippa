import torch
from typing import List,Dict, Tuple

from ._base._base_datamodule import BaseDataModule
from src.dl.utils import to_device


class CustomDataModule(BaseDataModule):
    def __init__(
            self,
            train_db_path: str,
            test_db_path: str,
            seg_targets: List[str],
            img_transforms: List[str],
            inst_transforms: List[str],
            normalize: bool=False,
            return_weight_map: bool=False,
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
            seg_targets (List[str]):
                A list of target map names needed for the loss cfunc.
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
            batch_size (int, default=8):
                Batch size for the dataloader
            num_workers (int, default=8):
                number of cpu cores/threads used in the dataloading 
                process.
        """        
        super().__init__(
            train_db_path,
            test_db_path,
            seg_targets,
            img_transforms,
            inst_transforms,
            normalize,
            return_weight_map,
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