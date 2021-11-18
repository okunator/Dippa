from pathlib import Path

MODULE_DIR = Path(__file__).parents[0].absolute()
PROJ_DIR = Path(__file__).parents[1].absolute()
CONF_DIR = Path(MODULE_DIR / "data/conf")
DATA_DIR = Path(PROJ_DIR / "datasets/data")
PATCH_DIR = Path(PROJ_DIR / "datasets/patches")
RESULT_DIR = Path(PROJ_DIR / "results")
EXPERIMENT_YML_PATH = Path(PROJ_DIR / "conf/experiment.yml")
MODEL_YML_PATH = Path(PROJ_DIR / "conf/conf_model.yml")
TRAIN_YML_PATH = Path(PROJ_DIR / "conf/conf_train.yml")
DATAMODULE_YML_PATH = Path(PROJ_DIR / "conf/conf_datamodule.yml")
TEST_MODEL_YML_PATH = Path(PROJ_DIR / "conf/_tests/conf_model_test.yml")
TEST_TRAIN_YML_PATH = Path(PROJ_DIR / "conf/_tests/conf_train_test.yml")
TEST_DATAMODULE_YML_PATH = Path(PROJ_DIR / "conf/_tests/conf_datamodule_test.yml")


__all__ = [
    "MODULE_DIR", "PROJ_DIR", "CONF_DIR", "DATA_DIR", "PATCH_DIR",
    "RESULT_DIR", "EXPERIMENT_YML_PATH", "MODEL_YML_PATH", "TRAIN_YML_PATH", 
    "TEST_MODEL_YML_PATH", "TEST_TRAIN_YML_PATH", "DATAMODULE_YML_PATH",
    "TEST_DATAMODULE_YML_PATH"
]

