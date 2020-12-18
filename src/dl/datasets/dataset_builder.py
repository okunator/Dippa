import src.img_processing.augmentations as augs
import src.dl.dataset as ds
from torch.utils.data import Dataset
from omegaconf import DictConfig
from src.utils.file_manager import ProjectFileManager

class DatasetBuilder(ProjectFileManager):
    def __init__(self,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig) -> None:
        """
        Takes in 

        Args:
            dataset_args (DictConfig): omegaconfig DictConfig specifying arguments related to the dataset
                that is being used. config.py for more info
            experiment_args (DictConfig): omegaconfig DictConfig specifying argumentst hat are used for 
                creating result folders and files. Check config.py for more info
            """
        super(DatasetBuilder, self).__init__(dataset_args, experiment_args)


    def get_augs(self, augs_list):
        """
        Compose the augmentations in config.py to a augmentation pipeline

        Args:
            aug_list (List[str]): List of augmentations specified in config.py
        """
        aug_list = [augs.__dict__[ds.AUGS_LOOKUP[aug_name]] for aug_name in augs_list] 
        return augs.compose(aug_list)


    @classmethod
    def set_dataset(cls, fname:str, conf: DictConfig, **kwargs) -> Dataset:
        """
        Set the dataset according to config.py params

        Args:
            conf (DictConfig): the config.py file
        """

        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        c = cls(dataset_args, experiment_args)
        augs = c.get_augs(c.augmentations)
        
