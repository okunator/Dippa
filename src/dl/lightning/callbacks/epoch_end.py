import torch
import pytorch_lightning as pl
from typing import Dict, Tuple


# TODO _epoch_end hooks don't take in outputs... This does not work yet
class EpochEndCallback(pl.Callback):
    def __init__(self) -> None:
        """
        Callback to log mean iou and accuracy at the end of each epoch
        """
        super().__init__()

    def _epoch_end(self, 
                   outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the metrics
        """
        accuracy = torch.stack([x["accuracy"] for x in outputs]).mean()
        miou = torch.stack([x["miou"] for x in outputs]).mean()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        return {
            "accuracy": accuracy,
            "miou": miou,
            "loss": loss
        }

    def on_training_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Dict[str, torch.Tensor],
        ) -> None:
        """
        Compute the mean accuracies and mious and log them
        """
        res = self._epoch_end(outputs)
        for logger in trainer.logger.experiment:
            logger.log(
                {
                    "train/acc_epoch": res["accuracy"],
                    "train/iou_epoch": res["miou"],
                    "global_step": trainer.global_step
                }
            )

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Dict[str, torch.Tensor],
        ) -> None:
        """
        Compute the mean accuracies and mious and log them
        """
        res = self._epoch_end(outputs)
        for logger in trainer.logger.experiment:
            logger.log(
                {
                    "val/acc_epoch": res["accuracy"],
                    "val/iou_epoch": res["miou"],
                    "global_step": trainer.global_step
                }
            )

    def on_test_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Tuple[torch.Tensor],
        ) -> None:
        """
        Compute the mean accuracies and mious and log them
        """
        res = self._epoch_end(outputs)
        for logger in trainer.logger.experiment:
            logger.log(
                {
                    "test/acc_epoch": res["accuracy"],
                    "test/iou_epoch": res["miou"],
                    "global_step": trainer.global_step
                }
            )