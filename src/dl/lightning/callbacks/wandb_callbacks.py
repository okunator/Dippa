import wandb
import torch
import pytorch_lightning as pl
from typing import Dict


class WandbImageCallback(pl.Callback):
    """
    Logs the inputs and outputs of the model
    at the first validation batch
    """
    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int,
        ) -> None:
        """
        
        """
        if batch_idx == 0:
            img = batch["image"].float().to(device="cpu")
            inst_target = batch["binary_map"].long().to(device="cpu")
            type_target = batch["type_map"].long().to(device="cpu")
            soft_insts = outputs["instances"].to(device="cpu")
            soft_types = outputs["types"].to(device="cpu")
            insts = torch.argmax(soft_insts, dim=1)
            types = torch.argmax(soft_types, dim=1)
            gts = torch.stack([inst_target, type_target], dim=1).long()
            preds = torch.stack([insts, types], dim=1).long()

            trainer.logger.experiment[1].log({
                "val/soft_insts": [
                    wandb.Image(si[1, ...], caption="soft inst masks") 
                    for si in soft_insts
                ],
                "val/soft_types": [
                    wandb.Image(st[i, ...], caption="Soft type masks") 
                    for st in soft_types
                    for i in range(st.shape[0])
                ],
                "val/preds": [
                    wandb.Image(pred[i, ...].float(), caption="Predictions") 
                    for pred in preds
                    for i in range(pred.shape[0])
                ],
                "val/GT": [
                    wandb.Image(gt[i, ...].float(), caption="Ground truths") 
                    for gt in gts
                    for i in range(gt.shape[0])
                ],
                "val/img": [wandb.Image(img, caption="Input img")],
                "global_step": trainer.global_step
            })

