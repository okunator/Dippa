from typing import List, Optional
from torch.utils.data import Dataset
from omegaconf import DictConfig

import src.dl.datasets as ds


class DataSetBuilder:
    def __init__(self,
                 model_args: DictConfig,
                 training_args: DictConfig,
                 **kwargs) -> None:
        """
        Initializes the train & test time datsets based on the experiment.yml

        Args:
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used.
            training_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments that are
                used for training a network. Contains list of augmentations
        """
        self.augs: List[str] = training_args.augmentations
        self.aux_branch = model_args.decoder_branches.aux
        self.ds_name = model_args.decoder_branches.aux_type if self.aux_branch else "unet"
        assert self.ds_name in ("hover", "dist", "contour", "unet", "basic")

    def get_augs(self, augs_list: Optional[List[str]] = None):
        """
        Compose the augmentations in config.py to a augmentation pipeline

        Args:
            augs_list (List[str], optional): 
                List of augmentations specified in config.py
        """
        aug_list = [ds.__dict__[ds.AUGS_LOOKUP[aug_name]]() for aug_name in augs_list] if augs_list else []
        aug_list.append(ds.to_tensor()) 
        return ds.compose(aug_list)

    @classmethod
    def set_train_dataset(cls,
                          model_args: DictConfig,
                          training_args: DictConfig,
                          fname: str,
                          augs_list: Optional[List[str]] = None) -> Dataset:
        """
        Init the train dataset.

        Args:
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used.
            training_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments that are
                used for training a network. Contains list of augmentations
            fname (str):
                path to the hdf5 database file
            augs_list (List[str], optional, default=None):    
                List of augmentations specified in config.py
        """
        c = cls(model_args, training_args)
        aug = c.get_augs(augs_list)
        return ds.__dict__[ds.DS_LOOKUP[c.ds_name]](fname=fname, transforms=aug)

    @classmethod
    def set_test_dataset(cls,   
                         model_args: DictConfig,
                         training_args: DictConfig, 
                         fname: str) -> Dataset:
        """
        Init the test dataset. 

        Args:
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used.
            training_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments that are
                used for training a network. Contains list of augmentations
            fname (str): 
                path to the hdf5 database file
        """
        c = cls(model_args, training_args)
        aug = c.get_augs()
        return ds.__dict__[ds.DS_LOOKUP[c.ds_name]](fname=fname, transforms=aug)