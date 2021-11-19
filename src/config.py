from copy import deepcopy
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Union, List

from src.settings import (
    MODEL_YML_PATH, TRAIN_YML_PATH, DATAMODULE_YML_PATH,
    TEST_MODEL_YML_PATH, TEST_TRAIN_YML_PATH, TEST_DATAMODULE_YML_PATH
)


def get_conf(path: Union[str, Path]) -> DictConfig:
    """
    Get the config .yml file from path
    
    Args:
    --------
        path (Union[str, Path]):
            Path to the .yml file

    Returns:
    --------    
        DictConfig: The config dict.
    """
    return OmegaConf.load(path)


def merge_conf(
        train_conf: DictConfig,
        data_conf: DictConfig,
        model_conf: DictConfig
    ) -> DictConfig:
    """
    Merge the different config files together and check for obivous
    conflicts.
    
    Args:
    ---------
        train_conf (DictConfig):
            Config dict for training args
        data_conf (DictConfig):
            Config dict for datamodule args
        model_conf (DictConfig):
            Config dict for model architecture specs
            
    Returns:
    ---------
        DictConfig: Merged Full config file.
    """
    br1 = sorted(train_conf["training"]["loss"]["edge_weights"].keys())
    br2 = sorted(train_conf["training"]["loss"]["class_weights"].keys())
    br3 = sorted(train_conf["training"]["loss"]["branch_losses"].keys())
    br4 = sorted(train_conf["training"]["metrics"].keys())
    br5 = sorted(model_conf["model"]["decoder"]["branches"].keys())

    br6 = deepcopy(data_conf["datamodule"]["targets"])
    if data_conf["datamodule"]["dataset_type"] in ("hover", "contour", "dist"):
        br6.append("aux")
    br6 = sorted(br6)

    if not br1 == br2 == br3 == br4 == br5 == br6: 
        raise ValueError(f"""
            Got mismatching branch keys. edge_weights: {br1}, class_weights: 
            {br2}, branch losses: {br3}, metrics {br4}, deocder branches: {br5}
            Data module out masks: {br6}"""
        )
        
    return OmegaConf.merge(train_conf, data_conf, model_conf)
    
    
    
MODEL_CONFIG = get_conf(MODEL_YML_PATH)
TRAIN_CONFIG = get_conf(TRAIN_YML_PATH)
DATA_CONFIG = get_conf(DATAMODULE_YML_PATH)
CONFIG = merge_conf(TRAIN_CONFIG, DATA_CONFIG, MODEL_CONFIG)

TEST_MODEL_CONFIG = get_conf(TEST_MODEL_YML_PATH)
TEST_TRAIN_CONFIG = get_conf(TEST_TRAIN_YML_PATH)
TEST_DATA_CONFIG = get_conf(TEST_DATAMODULE_YML_PATH)
TEST_CONFIG = merge_conf(TEST_TRAIN_CONFIG, TEST_DATA_CONFIG, TEST_MODEL_CONFIG)


__all__ = [
    "CONFIG", "MODEL_CONFIG", "TRAIN_CONFIG", "TEST_MODEL_CONFIG",
    "TEST_TRAIN_CONFIG", "TEST_CONFIG", "TEST_DATA_CONFIG", "DATA_CONFIG"
]