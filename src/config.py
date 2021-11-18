from omegaconf import OmegaConf
from copy import deepcopy
from src.settings import (
    MODEL_YML_PATH, TRAIN_YML_PATH, DATAMODULE_YML_PATH,
    TEST_MODEL_YML_PATH, TEST_TRAIN_YML_PATH, TEST_DATAMODULE_YML_PATH
)

MODEL_CONFIG = OmegaConf.load(MODEL_YML_PATH)
TRAIN_CONFIG = OmegaConf.load(TRAIN_YML_PATH)
DATA_CONFIG = OmegaConf.load(DATAMODULE_YML_PATH)

br1 = sorted(TRAIN_CONFIG["training"]["loss"]["edge_weights"].keys())
br2 = sorted(TRAIN_CONFIG["training"]["loss"]["class_weights"].keys())
br3 = sorted(TRAIN_CONFIG["training"]["loss"]["branch_losses"].keys())
br4 = sorted(TRAIN_CONFIG["training"]["metrics"].keys())
br5 = sorted(MODEL_CONFIG["model"]["decoder"]["branches"].keys())

br6 = deepcopy(DATA_CONFIG["datamodule"]["targets"])
if DATA_CONFIG["datamodule"]["dataset_type"] in ("hover", "contour", "dist"):
    br6.append("aux")
br6 = sorted(br6)

assert br1 == br2 == br3 == br4 == br5 == br6, (
    f"Got mismatching branch keys. edge_weights: {br1}, class_weights: {br2}",
    f"branch losses: {br3}, metrics {br4}, deocder branches: {br5}. Data module",
    f" out masks: {br6}"
)

CONFIG = OmegaConf.merge(MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG)

TEST_MODEL_CONFIG = OmegaConf.load(TEST_MODEL_YML_PATH)
TEST_TRAIN_CONFIG = OmegaConf.load(TEST_TRAIN_YML_PATH)
TEST_DATA_CONFIG = OmegaConf.load(TEST_DATAMODULE_YML_PATH)
TEST_CONFIG = OmegaConf.merge(
    TEST_MODEL_CONFIG, TEST_TRAIN_CONFIG, TEST_DATA_CONFIG
)


__all__ = [
    "CONFIG", "MODEL_CONFIG", "TRAIN_CONFIG", "TEST_MODEL_CONFIG",
    "TEST_TRAIN_CONFIG", "TEST_CONFIG", "TEST_DATA_CONFIG", "DATA_CONFIG"
]