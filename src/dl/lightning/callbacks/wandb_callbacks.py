import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional

try:
    import wandb
except ImportError as e:
    raise ImportError("wandb required. `pip install wandb`")


class WandbImageCallback(pl.Callback):
    def __init__(
            self,
            classes: Dict[str, int],
            sem_classes: Optional[Dict[str, int]]
        ) -> None:
        """
        Callback that logs prediction masks to wandb
        """
        super(WandbImageCallback, self).__init__()
        self.classes = {v: k for k, v in classes.items()} # flip

        if sem_classes is not None:
            self.sem_classes = {v: k for k, v in sem_classes.items()}

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
        if batch_idx in (0, 1, 2):
            log_dict = {
                "global_step": trainer.global_step,
                "epoch": trainer.current_epoch
            }
            img = batch["image"].detach().to("cpu").numpy()

            if "type_map" in list(batch.keys()):
                type_target = batch["type_map"].detach().to("cpu").numpy()
                soft_types = outputs["type_map"].detach().to("cpu")
                types = torch.argmax(F.softmax(soft_types, dim=1), dim=1).numpy()

                log_dict["val/cell_types"] = [
                    wandb.Image(
                        im.transpose(1, 2, 0),
                        masks = {
                            "predictions": {
                                "mask_data": t,
                                "class_labels": self.classes
                            },
                            "ground_truth": {
                                "mask_data": tt,
                                "class_labels": self.classes
                            }
                        }
                    )
                    for im, t, tt in zip(img, types, type_target)
                ]
            
            if "sem_map" in list(batch.keys()):
                sem_target = batch["sem_map"].detach().to("cpu").numpy()
                soft_sem = outputs["sem_map"].detach().to(device="cpu")
                sem = torch.argmax(F.softmax(soft_sem, dim=1), dim=1).numpy()

                log_dict["val/cell_areas"] = [
                    wandb.Image(
                        im.transpose(1, 2, 0),
                        masks = {
                            "predictions": {
                                "mask_data": s,
                                "class_labels": self.sem_classes
                            },
                            "ground_truth": {
                                "mask_data": st,
                                "class_labels": self.sem_classes
                            }
                        }
                    )
                    for im, s, st in zip(img, sem, sem_target)
                ]

            if "aux_map" in list(batch.keys()):
                aux = outputs["aux_map"].detach().to(device="cpu")
                log_dict["val/aux_maps"] = [
                    wandb.Image(a[i, ...], caption="Aux maps") 
                    for a in aux
                    for i in range(a.shape[0])
                ]

            trainer.logger.experiment[1].log(log_dict)

