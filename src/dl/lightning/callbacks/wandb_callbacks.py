import wandb
import torch
import pytorch_lightning as pl
from typing import Dict


class WandbImageCallback(pl.Callback):

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
        Logs the inputs and outputs of the model
        at the first validation batch to weights and biases.
        """
        if batch_idx == 0:
            img = batch["image"].float().to(device="cpu")
            soft_insts = outputs["instances"].to(device="cpu")
            inst_target = batch["binary_map"].long().to(device="cpu")
            insts = torch.argmax(soft_insts, dim=1)

            log_dict = {
                "val/soft_insts": [
                    wandb.Image(si[1, ...], caption="soft inst masks") 
                    for si in soft_insts
                ],
                "val/inst_preds": [
                    wandb.Image(pred.float(), caption="Predictions") 
                    for pred in insts
                ],
                "val/inst_GT": [
                    wandb.Image(gt.float(), caption="Inst ground truths") 
                    for gt in inst_target
                ],
                "val/img": [wandb.Image(img, caption="Input img")],
                "global_step": trainer.global_step
            }


            if "type_map" in list(batch.keys()):
                type_target = batch["type_map"].long().to(device="cpu")
                soft_types = outputs["types"].to(device="cpu")
                types = torch.argmax(soft_types, dim=1)

                log_dict["val/soft_types"] = [
                    wandb.Image(st[i, ...], caption="Soft type masks") 
                    for st in soft_types
                    for i in range(st.shape[0])
                ]

                log_dict["val/type_GT"] = [
                    wandb.Image(gt.float(), caption="Type ground truths") 
                    for gt in type_target
                ]

                log_dict["val/type_pred"] = [
                    wandb.Image(pred.float(), caption="Type ground truths") 
                    for pred in types
                ]

            if "sem_map" in list(batch.keys()):
                sem_target = batch["sem_map"].long().to(device="cpu")
                soft_sem = outputs["sem"].to(device="cpu")
                sem = torch.argmax(soft_sem, dim=1)

                log_dict["val/soft_sem"] = [
                    wandb.Image(ss[i, ...], caption="Soft semantic masks") 
                    for ss in soft_sem
                    for i in range(ss.shape[0])
                ]

                log_dict["val/sem_GT"] = [
                    wandb.Image(gt.float(), caption="Semantic ground truths") 
                    for gt in sem_target
                ]

                log_dict["val/sem_pred"] = [
                    wandb.Image(pred.float(), caption="Sem ground truths") 
                    for pred in sem
                ]


            if "aux_map" in list(batch.keys()):
                aux = outputs["aux"].to(device="cpu")
                log_dict["val/aux_maps"] = [
                    wandb.Image(a[i, ...], caption="Aux maps") 
                    for a in aux
                    for i in range(a.shape[0])
                ]

            trainer.logger.experiment[1].log(log_dict)

