from omegaconf import OmegaConf
from src.settings import (
    MODEL_YML_PATH, TRAIN_YML_PATH, TEST_MODEL_YML_PATH, TEST_TRAIN_YML_PATH
)

MODEL_CONFIG = OmegaConf.load(MODEL_YML_PATH)
TRAIN_CONFIG = OmegaConf.load(TRAIN_YML_PATH)

br1 = TRAIN_CONFIG["training"]["loss"]["edge_weights"].keys()
br2 = TRAIN_CONFIG["training"]["loss"]["class_weights"].keys()
br3 = TRAIN_CONFIG["training"]["loss"]["branch_losses"].keys()
br4 = TRAIN_CONFIG["training"]["metrics"].keys()
br5 = MODEL_CONFIG["model"]["decoder"]["branches"].keys()

assert br1 == br2 == br3 == br4 == br5, (
    f"Got mismatching branch keys. edge_weights: {br1}, class_weights: {br2}",
    f"branch losses: {br3}, metrics {br4}, deocder branches: {br5}"
)

CONFIG = OmegaConf.merge(MODEL_CONFIG, TRAIN_CONFIG)

TEST_MODEL_CONFIG = OmegaConf.load(TEST_MODEL_YML_PATH)
TEST_TRAIN_CONFIG = OmegaConf.load(TEST_TRAIN_YML_PATH)
TEST_CONFIG = OmegaConf.merge(TEST_MODEL_CONFIG, TEST_TRAIN_CONFIG)


__all__ = [
    "CONFIG", "MODEL_CONFIG", "TRAIN_CONFIG", "TEST_MODEL_CONFIG",
    "TEST_TRAIN_CONFIG", "TEST_CONFIG"
]