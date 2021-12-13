import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import List

from . import *


dat_modules = vars()


def prepare_datamodule(
        name: str,
        conf: DictConfig=None,
        seg_targets: List[str]=["inst", "type"],
        img_transforms: List[str]=["hue_sat", "non_rigid", "blur"],
        inst_transforms: List[str]=["hover"],
        normalize: bool=False,
        return_weight_map: bool=False,
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
            
    Returns:
    ---------
        pl.LightningDataModule: Initialized lightning DataModule
    """
    if conf is not None:
        kwargs["seg_targets"] = conf.datamodule.seg_targets
        kwargs["img_transforms"] = conf.datamodule.img_transforms
        kwargs["inst_transforms"] = conf.datamodule.inst_transforms
        kwargs["return_weight_map"] = conf.datamodule.weight_map
        kwargs["normalize"] = conf.datamodule.normalize_input
        kwargs["batch_size"] = conf.datamodule.batch_size
        kwargs["num_workers"] = conf.datamodule.num_workers
    else:
        kwargs["seg_targets"] = seg_targets
        kwargs["img_transforms"] = img_transforms
        kwargs["inst_transforms"] = inst_transforms
        kwargs["return_weight_map"] = return_weight_map
        kwargs["normalize"] = normalize
        kwargs["batch_size"] = batch_size
        kwargs["num_workers"] = num_workers

    allowed = dat_modules["DATAMODULE_LOOKUP"].keys()
    assert name in allowed, (
        f"Illegal datamodule name given. Allowed: {allowed}. Got {name}."
    )
    
    key = dat_modules["DATAMODULE_LOOKUP"][name]
    
    return dat_modules[key](**kwargs)
    