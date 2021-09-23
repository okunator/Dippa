import importlib
import albumentations as A
from typing import List, Optional
from torch.utils.data import Dataset


ds = importlib.import_module("src.data.datasets")


class DatasetBuilder:
    def __init__(self,
                 decoder_aux_branch: str=None,
                 **kwargs) -> None:
        """
        Initializes the train & test time datsets based on the experiment.yml

        Args:
        -----------
            deocder_aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet dataset
                is used in the dataloader.
        """

        self.ds_name = "unet"
        if decoder_aux_branch is not None:
            self.ds_name = decoder_aux_branch

        assert self.ds_name in ("hover", "dist", "contour", "unet", "basic")

    def get_augs(self, augs_list: Optional[List[str]]=None) -> A.Compose:
        """
        Compose the augmentations in config.py to a augmentation pipeline

        Args:
        -----------
            augs_list (List[str], optional, default=None): 
                List of augmentations specified in config.py. 
                If None, then no augmentations are used

        Returns:
        -----------
            A.Compose object containing the albumentations augmentations.
        """
        kwargs={"height":256, "width":256}
        aug_list = [ds.__dict__[ds.AUGS_LOOKUP[aug_name]](**kwargs) for aug_name in augs_list] if augs_list else []
        aug_list.append(ds.to_tensor()) 
        return ds.compose(aug_list)

    @classmethod
    def set_train_dataset(cls,
                          fname: str,
                          augmentations: List[str]=None,
                          decoder_aux_branch: str=None,
                          normalize_input: bool=True) -> Dataset:
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
            decoder_aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet dataset
                is used in the dataloader.
            normalize_input (bool, default=True):
                If True, channel-wise normalization for the input images is
                applied.

        Returns:
        ------------
            torch.utils.data.Dataset. Initialized torch Dataset object 
        """
        c = cls(decoder_aux_branch)
        aug = c.get_augs(augmentations)
        return ds.__dict__[ds.DS_LOOKUP[c.ds_name]](fname=fname, transforms=aug, normalize_input=normalize_input)

    @classmethod
    def set_test_dataset(cls,
                         fname: str,
                         decoder_aux_branch: str=None,
                         normalize_input: bool=True) -> Dataset:
        """
        Init the test dataset. No augmentations used. Only ndarray to tensor 
        conversion.

        Args:
        ------------
            fname (str): 
                path to the database file
            decoder_aux_branch (str, default=None):
                The type of the auxiliary branch. If None, the unet dataset
                is used in the dataloader.
            normalize_input (bool, default=True):
                If True, channel-wise normalization for the input images is 
                applied.

        Returns:
        ------------
            torch.utils.data.Dataset. Initialized torch Dataset object 
        """
        c = cls(decoder_aux_branch)
        aug = c.get_augs()
        return ds.__dict__[ds.DS_LOOKUP[c.ds_name]](fname=fname, transforms=aug, normalize_input=normalize_input)