from pathlib import Path

MODULE_DIR = Path(__file__).parents[0].absolute()
PROJ_DIR = Path(__file__).parents[1].absolute()
CONF_DIR = Path(MODULE_DIR / "conf")
DATA_DIR = Path(PROJ_DIR / "datasets/data")
PATCH_DIR = Path(PROJ_DIR / "datasets/patches")
RESULT_DIR = Path(PROJ_DIR / "results")
EXPERIMENT_YML_PATH = Path(PROJ_DIR / "experiment.yml")

