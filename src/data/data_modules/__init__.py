from .pannuke_datamodule import PannukeDataModule
from .consep_datamodule import ConsepDataModule
from .custom_datamodule import CustomDataModule


DATAMODULE_LOOKUP = {
    "pannuke": "PannukeDataModule",
    "consep": "ConsepDataModule",
    "custom": "CustomDataModule"
}


__all__ = [
    "DATAMODULE_LOOKUP", "PannukeDataModule",
    "ConsepDataModule", "CustomDataModule"
]