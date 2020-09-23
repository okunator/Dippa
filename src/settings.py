from pathlib import Path

MODULE_DIR = Path(__file__).parents[0].absolute()
PROJ_DIR = Path(__file__).parents[1].absolute()
CONF_DIR = Path(MODULE_DIR / "conf")
DATA_DIR = Path(PROJ_DIR / "datasets")
PATCH_DIR = Path(PROJ_DIR / "patches")
RESULT_DIR = Path(PROJ_DIR / "results/tests")

