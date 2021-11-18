import pytest
import pytorch_lightning as pl

from src.settings import MODULE_DIR
from src.config import TEST_CONFIG

from src.dl.models import MultiTaskSegModel
from src.dl.lightning import SegTrainer, SegExperiment
from src.data.data_modules.utils import prepare_datamodule


@pytest.fixture
def datamodule() -> pl.LightningDataModule:
    train_db_path = MODULE_DIR / "data/datasets/_tests/tiny_test.h5"
    test_db_path = MODULE_DIR / "data/datasets/_tests/tiny_test.h5"
    
    return prepare_datamodule(
        name="custom",
        train_db_path=train_db_path,
        test_db_path=test_db_path,
        conf=TEST_CONFIG
    )
    
@pytest.fixture
def model() -> pl.LightningModule:
    model = MultiTaskSegModel.from_conf(TEST_CONFIG)
    lightning_model = SegExperiment.from_conf(
        model, TEST_CONFIG, hparams_to_yaml=False
    )
    
    return lightning_model

def test_lightning_steps(datamodule, model) -> None:
    extra_callbacks = []
    trainer = SegTrainer.from_conf(
        conf=TEST_CONFIG,
        extra_callbacks=extra_callbacks,
        detect_anomaly=True,
        fast_dev_run=True,
        log_every_n_steps=1,
        accelerator="cpu"
    )
    
    trainer.fit(model=model, datamodule=datamodule)