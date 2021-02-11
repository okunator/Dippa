import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Ralamb, RAdam, Lookahead

from src.utils.file_manager import FileManager
from src.dl.datasets.dataset_builder import DatasetBuilder
from src.dl.optimizers.optim_builder import OptimizerBuilder
from src.dl.losses.loss_builder import LossBuilder
from src.dl.torch_utils import to_device, argmax_and_flatten, iou, accuracy


class SegModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 experiment_args: DictConfig,
                 dataset_args: DictConfig,
                 model_args: DictConfig,
                 training_args: DictConfig,
                 runtime_args: DictConfig,
                 **kwargs) -> None:
        """
        Pytorch lightning model wrapper.
        Uses the experiment.yml file.

        Args:
            model (nn.Module):
                pytorch model specification.
            experiment_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments that
                are used for creating result folders and files. 
            dataset_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the dataset that is being used.
            model_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments related 
                to the model architecture that is being used.
            training_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying arguments that are
                used for training a network.
            runtime_args (omegaconf.DictConfig): 
                Omegaconf DictConfig specifying batch size and 
                input image size for the model.
        """
        super(SegModel, self).__init__()
        self.model: nn.Module = model

        # General args
        self.augmentations: List[str] = training_args.augmentations
        self.batch_size: int = runtime_args.batch_size
        self.input_size: int = runtime_args.model_input_size

        # model args
        self.type_branch: bool = model_args.decoder_branches.type
        self.aux_branch: bool = model_args.decoder_branches.aux
        self.aux_type: str = model_args.decoder_branches.aux_type

        # Loss args
        self.inst_branch_loss: str = training_args.loss_args.inst_branch_loss
        self.type_branch_loss: str = training_args.loss_args.inst_branch_loss
        self.aux_branch_loss: str = training_args.loss_args.inst_branch_loss
        self.edge_weight: str = training_args.loss_args.edge_weight
        self.class_weights: str = training_args.loss_args.class_weights

        # Optimizer args
        self.optimizer_name: str = training_args.optimizer_args.optimizer
        self.lr: float = training_args.optimizer_args.lr
        self.encoder_lr: float = training_args.optimizer_args.encoder_lr 
        self.weight_decay: float = training_args.optimizer_args.weight_decay 
        self.encoder_weight_decay: float = training_args.optimizer_args.encoder_weight_decay
        self.scheduler_factor: float = training_args.optimizer_args.scheduler_factor
        self.schduler_patience: int = training_args.optimizer_args.schduler_patience
        self.lookahead: bool = training_args.optimizer_args.lookahead
        self.bias_weight_decay: bool = training_args.optimizer_args.bias_weight_decay
        self.save_hyperparameters()

        # init file manager
        self.fm = FileManager(
            experiment_args=experiment_args,
            dataset_args=dataset_args
        )

        # init loss function
        self.criterion = LossBuilder.set_loss(
            decoder_branches=model_args.decoder_branches,
            loss_args=training_args.loss_args,
            binary_weights=self.binary_class_weights,
            type_weights=self.type_branch_loss
        )

        # init optimizer
        self.optimizer = OptimizerBuilder.set_optimizer(
            training_args.optimizer_args, self.model
        )

    @classmethod
    def from_conf(cls, conf: DictConfig, model: nn.Module):
        model = model
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        model_args = conf.model_args
        training_args = conf.training_args
        runtime_args = conf.runtime_args
        
        return cls(
            model,
            experiment_args,
            dataset_args,
            model_args,
            training_args,
            runtime_args
        )

    @property
    def train_data(self):
        #TODO npy files
        return self.fm.databases['train'][self.input_size]
    
    @property
    def valid_data(self):
        # TODO npy files
        return self.fm.databases['valid'][self.input_size]
    
    @property
    def test_data(self):
        # TODO npy files
        return self.fm.databases['test'][self.input_size]

    @property
    def binary_class_weights(self):
        """
        Compute weights for fg and bg from training data pixels
        """
        weights = None
        if self.class_weights:
            num_class_pixels = self.fm.get_class_pixels(self.train_data.as_posix())
            num_binary_pixels = np.sum(num_class_pixels, where=[False] + [True]*(len(num_class_pixels)-1))
            class_pixels = np.array([num_class_pixels[0], num_binary_pixels])
            weights = to_device(1-class_pixels/class_pixels.sum())
        return weights

    @property
    def type_class_weights(self):
        """
        Compute weights for type classes from training data pixels
        """
        weights = None
        if self.class_weights:
            class_pixels = self.fm.get_class_pixels(self.train_data.as_posix())
            weights = to_device(1-class_pixels/class_pixels.sum())
        return weights

    # Lightning framework stuff:
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        """
        # Get data
        x = batch["image"].float()
        inst_target = batch["binary_map"].long()
        target_weight = batch["weight_map"].float()
            
        type_target = None
        if self.type_branch:
            type_target = batch["type_map"].long()

        aux_target = None
        if self.aux_branch:
            if self.aux_type == "hover":
                xmap = batch["xmap"].float()
                ymap = batch["ymap"].float()
                aux_target = torch.stack([xmap, ymap], dim=1)
            # TODO: other aux branches

        # Forward pass
        soft_mask = self.forward(x)

        # Compute loss
        loss = self.criterion(
            yhat_inst=soft_mask["instances"], 
            target_inst=inst_target, 
            yhat_type=soft_mask["types"],
            target_type=type_target, 
            yhat_aux=soft_mask["aux"],
            target_aux=aux_target,
            target_weight=target_weight,
            edge_weight=1.1
        )

        # Compute metrics for monitoring
        key = "types" if self.type_branch else "instances" 
        type_acc = accuracy(soft_mask[key], type_target, "softmax")
        type_iou = iou(soft_mask[key], type_target, "softmax")

        return {
            "loss":loss,
            "accuracy":type_acc.mean(),
            "mean_iou":type_iou.mean()
        }

    def step_return_dict(self, z: torch.Tensor, phase: str) -> Dict[str, torch.Tensor]:
        """
        Batch level metrics
        """
        logs = {
            f"{phase}_loss": z["loss"],
            f"{phase}_accuracy": z["accuracy"],
            f"{phase}_mean_iou": z["mean_iou"]
        }

        return {
            "loss": z["loss"],
            "accuracy": z["accuracy"],
            "mean_iou": z["mean_iou"],
            "log": logs
        }

    def epoch_end(self, outputs: torch.Tensor, phase: str) -> Dict[str, torch.Tensor]:
        """
        Full train data metrics
        """
        accuracy = torch.stack([x["accuracy"] for x in outputs]).mean()
        iou = torch.stack([x["mean_iou"] for x in outputs]).mean()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            f"avg_{phase}_loss": loss,
            f"avg_{phase}_accuracy": accuracy,
            f"avg_{phase}_iou": iou,
        }

        return {
            "loss": loss,
            "log": logs,
        }
    
    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        z = self.step(train_batch, batch_idx)
        return_dict = self.step_return_dict(z, "train")
        return return_dict

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        z = self.step(val_batch, batch_idx)
        return_dict = self.step_return_dict(z, "val")
        return return_dict

    def test_step(self, test_batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        z = self.step(test_batch, batch_idx)
        return_dict = self.step_return_dict(z, "test")
        return return_dict
    
    def training_epoch_end(self, outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return_dict = self.epoch_end(outputs, "train")
        return return_dict

    def validation_epoch_end(self, outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return_dict = self.epoch_end(outputs, "val")
        return return_dict

    def test_epoch_end(self, outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return_dict = self.epoch_end(outputs, "test")
        return return_dict

    def configure_optimizers(self):
        optimizer = self.optimizer

        # Scheduler
        scheduler = [{
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer[0], factor=self.factor, patience=self.patience
            ),
            'monitor': 'avg_val_loss',
            'interval': 'epoch',
            'frequency': 1
        }]

        return optimizer, scheduler 

    def prepare_data(self):
        self.trainset = DatasetBuilder.set_train_dataset(
            fname=self.train_data.as_posix(), 
            preproc_style=self.fm.preproc_style,
            augs_list=self.fm.train_augs
        )
        self.validset = DatasetBuilder.set_test_dataset(
            fname=self.valid_data.as_posix(), 
            preproc_style=self.fm.preproc_style,
        )
        self.testset = DatasetBuilder.set_test_dataset(
            fname=self.test_data.as_posix(), 
            preproc_style=self.fm.preproc_style,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8
        )
