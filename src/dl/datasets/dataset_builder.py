import src.img_processing.augmentations as augs
import src.dl.datasets as ds
from typing import List, Optional
from torch.utils.data import Dataset
from omegaconf import DictConfig


class DatasetBuilder:
    def __init__(self, preproc_style: str) -> None:
        """
        Class to initialize train and test dataset for the lightning model

        Args:
            preproc_style (str): 
                One of ("basic", "hover", "unet", "micro")
        """
        self.preproc_style = preproc_style

    def get_augs(self, augs_list: Optional[List[str]] = None):
        """
        Compose the augmentations in config.py to a augmentation pipeline

        Args:
            augs_list (List[str], optional): 
                List of augmentations specified in config.py
        """
        
        aug_list = [augs.__dict__[ds.AUGS_LOOKUP[aug_name]]() for aug_name in augs_list] if augs_list else []
        aug_list.append(augs.to_tensor()) 
        return augs.compose(aug_list)

    @classmethod
    def set_train_dataset(cls, 
                          fname: str,
                          preproc_style: str = "basic", 
                          augs_list: Optional[List[str]] = None,
                          **kwargs) -> Dataset:
        """
        Init the train dataset. Chooses the data pre-processing style and augmentations from config.py 

        Args:
            fname (str):
                path to the hdf5 database file
            preproc_style (str): 
                One of ("basic", "hover", "unet", "micro")
            augs_list (List[str], optional):    
                List of augmentations specified in config.py
        """
        c = cls(preproc_style)
        aug = c.get_augs(augs_list)
        return ds.__dict__[ds.DS_LOOKUP[preproc_style]](fname=fname, transforms=aug)

    @classmethod
    def set_test_dataset(cls, 
                         fname: str,
                         preproc_style: str = "basic", 
                         **kwargs) -> Dataset:
        """
        Init the test dataset. Chooses the data pre-processing style from config.py but data augmentation 
        is ignored.  

        Args:
            fname (str): 
                path to the hdf5 database file
            preproc_style (str): 
                One of ("basic", "hover", "unet", "micro")
        """
        c = cls(preproc_style)
        aug = c.get_augs()
        return ds.__dict__[ds.DS_LOOKUP[c.preproc_style]](fname=fname, transforms=aug)
        
