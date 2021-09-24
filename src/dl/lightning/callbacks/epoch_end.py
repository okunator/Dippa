import torch
import pytorch_lightning as pl


class EpochEndCallback(pl.Callback):
    def __init__(self) -> None:
        """
        ddd
        """
        super().__init__()

    def on_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule
        ) -> None:
        """
        dddd
        """
        raise NotImplementedError