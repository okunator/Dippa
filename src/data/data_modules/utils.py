import pytorch_lightning as pl
from omegaconf import OmegaConf
from typing import List

from . import *


dat_modules = vars()


def prepare_datamodule(
        name: str,
        conf: OmegaConf=None,
        target_types: List[str]=None,
        dataset_type: str="hover",
        augs: List[str]=["hue_sat", "non_rigid", "blur"],
        normalize: bool=False,
        return_weight_map: bool=False,
        rm_touching_nuc_borders: bool=False,
        batch_size: int=8,
        num_workers: int=8,
        **kwargs
    ) -> pl.LightningDataModule:
    """
    Prepare a pl.lightning datamodule based on given args
    
    Args:
    ---------
        name (str):
            Name of the datamodule. One of: "custom", "pannuke", "consep"
        conf (OmegaConf, default=None):
            OmegaConf config file specifying the datamodule params. If
            None, the kwargs are used to init the datamodule.
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
        
            
    Returns:
    ---------
        pl.LightningDataModule: Initialized lightning DataModule
    """
    if conf is not None:
        kwargs["target_types"] = conf.datamodule.targets
        kwargs["dataset_type"] = conf.datamodule.dataset_type
        kwargs["augs"] = conf.datamodule.augmentations
        kwargs["return_weight_map"] = conf.datamodule.weight_map
        kwargs["normalize"] = conf.datamodule.normalize_input
        kwargs["rm_touching_nuc_borders"] = conf.datamodule.rm_overlaps
        kwargs["batch_size"] = conf.datamodule.batch_size
        kwargs["num_workers"] = conf.datamodule.num_workers
    else:
        kwargs["target_types"] = target_types
        kwargs["dataset_type"] = dataset_type
        kwargs["augs"] = augs
        kwargs["return_weight_map"] = return_weight_map
        kwargs["normalize"] = normalize
        kwargs["rm_touching_nuc_borders"] = rm_touching_nuc_borders
        kwargs["batch_size"] = batch_size
        kwargs["num_workers"] = num_workers

    allowed = dat_modules["DATAMODULE_LOOKUP"].keys()
    assert name in allowed, (
        f"Illegal datamodule name given. Allowed: {allowed}. Got {name}."
    )
    
    key = dat_modules["DATAMODULE_LOOKUP"][name]
    
    return dat_modules[key](**kwargs)
    