import pytest
from src.data.data_modules.utils import prepare_datamodule
from src.settings import MODULE_DIR
from src.config import TEST_CONFIG


@pytest.mark.parametrize("name", ["custom"])
def test_datamodule(name: str) -> None:
    """
    Test dataloading from a datamodule.
    """
    data_kwargs = {}
    data_kwargs["name"] = name
    data_kwargs["train_db_path"] = MODULE_DIR / "data/datasets/_tests/tiny_test.h5"
    data_kwargs["test_db_path"] = MODULE_DIR / "data/datasets/_tests/tiny_test.h5"

    dm = prepare_datamodule(
        **data_kwargs,
        conf=TEST_CONFIG
    )

    dm.setup()
    
    next(iter(dm.trainset))