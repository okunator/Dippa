from typing import List, Optional
from torch.utils.data import Dataset

import src.dl.datasets as ds


class DatasetBuilder:
    def __init__(self,
                 augmentations: List[str],
                 aux_branch: str=None,
                 **kwargs) -> None:
        """
        Initializes the train & test time datsets based on the experiment.yml

        Args:
        -----------
            augmentations (List[str]): 
                List of augmentations to be used for training
                One of ("rigid", "non_rigid", "hue_sat", "blur", "non_spatial",
                "random_crop", "center_crop", "resize")
            aux_branch_type (str, default=None):
                The type of the auxiliary branch. If None, the unet dataset
                is used in the dataloader.
        """
        self.augs: List[str] = augmentations
        self.ds_name: str = aux_branch if aux_branch is not None else "unet"
        assert self.ds_name in ("hover", "dist", "contour", "unet", "basic")

    def get_augs(self, augs_list: Optional[List[str]] = None):
        """
        Compose the augmentations in config.py to a augmentation pipeline

        Args:
        -----------
            augs_list (List[str], optional): 
                List of augmentations specified in config.py
        """
        kwargs={"height":256, "width":256}
        aug_list = [ds.__dict__[ds.AUGS_LOOKUP[aug_name]](**kwargs) for aug_name in augs_list] if augs_list else []
        aug_list.append(ds.to_tensor()) 
        return ds.compose(aug_list)

    @classmethod
    def set_train_dataset(cls,
                          fname: str,
                          augmentations: List[str]=None,
                          aux_branch: str=None,
                          norm: bool=True) -> Dataset:
        """
        Init the train dataset.

        Args:
        ------------
            fname (str):
                path to the database file
            augmentations (List[str]): 
                List of augmentations to be used for training
                One of ("rigid", "non_rigid", "hue_sat", "blur", "non_spatial",
                "random_crop", "center_crop", "resize")
            aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet dataset
                is used in the dataloader.
        """
        c = cls(augmentations, aux_branch)
        aug = c.get_augs(c.augs)
        return ds.__dict__[ds.DS_LOOKUP[c.ds_name]](fname=fname, transforms=aug, norm=norm)

    @classmethod
    def set_test_dataset(cls,
                         fname: str,
                         augmentations: List[str]=None,
                         aux_branch: str=None,
                         norm: bool=True) -> Dataset:
        """
        Init the test dataset. No augmentations used. Only ndarray to tensor conversion.

        Args:
        ------------
            fname (str): 
                path to the database file
            augmentations (List[str]): 
                List of augmentations to be used for training
                One of ("rigid", "non_rigid", "hue_sat", "blur", "non_spatial",
                "random_crop", "center_crop", "resize")
            aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet dataset
                is used in the dataloader.
        """
        c = cls(augmentations, aux_branch)
        aug = c.get_augs()
        return ds.__dict__[ds.DS_LOOKUP[c.ds_name]](fname=fname, transforms=aug, norm=norm)