import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl

from typing import List, Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.utils.file_manager import FileManager
from src.dl.datasets.dataset_builder import DatasetBuilder
from src.dl.optimizers.optim_builder import OptimizerBuilder
from src.dl.losses.loss_builder import LossBuilder
from src.dl.models.model_builder import Model
from src.dl.torch_utils import to_device, iou, accuracy


class SegModel(pl.LightningModule):
    def __init__(self,
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

        # Init all args for save_hyperparameters() (hypereparam logging)
        # General runtime args and augs
        self.augs: List[str] = training_args.augmentations
        self.batch_size: int = runtime_args.batch_size
        self.input_size: int = runtime_args.model_input_size
        self.num_workers: int = runtime_args.num_workers

        # Module args
        self.activation: str = model_args.architecture_design.module_args.activation
        self.normalization: str = model_args.architecture_design.module_args.normalization
        self.weight_standardize: bool = model_args.architecture_design.module_args.weight_standardize
        self.weight_init: str = model_args.architecture_design.module_args.weight_init

        # Encoder args
        self.in_channels: int = model_args.architecture_design.encoder_args.in_channels
        self.encoder_name: str = model_args.architecture_design.encoder_args.encoder
        self.pretrain: bool = model_args.architecture_design.encoder_args.pretrain
        self.depth: int = model_args.architecture_design.encoder_args.encoder_depth

        # Decoder_args
        self.n_blocks: int = model_args.architecture_design.decoder_args.n_blocks
        self.short_skips: str = model_args.architecture_design.decoder_args.short_skips
        self.long_skips: str = model_args.architecture_design.decoder_args.long_skips
        self.merge_policy: str = model_args.architecture_design.decoder_args.merge_policy
        self.upsampling: str = model_args.architecture_design.decoder_args.upsampling

        # Multi-tasking args
        self.type_branch: bool = model_args.decoder_branches.type
        self.aux_branch: bool = model_args.decoder_branches.aux
        self.aux_type: str = model_args.decoder_branches.aux_type

        # Loss args
        self.inst_branch_loss: str = training_args.loss_args.inst_branch_loss
        self.type_branch_loss: str = training_args.loss_args.inst_branch_loss
        self.aux_branch_loss: str = training_args.loss_args.inst_branch_loss
        self.edge_weight: bool = training_args.loss_args.edge_weight
        self.class_weights: bool = training_args.loss_args.class_weights

        # Optimizer args
        self.optimizer_name: str = training_args.optimizer_args.optimizer
        self.lr: float = training_args.optimizer_args.lr
        self.encoder_lr: float = training_args.optimizer_args.encoder_lr 
        self.weight_decay: float = training_args.optimizer_args.weight_decay 
        self.encoder_weight_decay: float = training_args.optimizer_args.encoder_weight_decay
        self.scheduler_factor: float = training_args.optimizer_args.scheduler_factor
        self.scheduler_patience: int = training_args.optimizer_args.scheduler_patience
        self.lookahead: bool = training_args.optimizer_args.lookahead
        self.bias_weight_decay: bool = training_args.optimizer_args.bias_weight_decay

        # save args to a file
        self.save_hyperparameters()

        # init file manager
        self.fm = FileManager(
            experiment_args=experiment_args,
            dataset_args=dataset_args
        )

        # database paths
        self.db_dict = self.fm.get_databases(self.fm.train_dataset)
        self.train_data = self.db_dict['train'][self.input_size]
        self.valid_data = self.db_dict['valid'][self.input_size]
        self.test_data = self.db_dict['test'][self.input_size]

        # init model
        if self.aux_branch:
            self.aux_channels = 2 if self.aux_type == "hover" else 1

        self.model = Model(model_args, n_classes=self.fm.n_classes, aux_out_channels=self.aux_channels)

        # Redundant but necessary for experiment logging..
        self.optimizer_args = training_args.optimizer_args
        self.decoder_branch_args = model_args.decoder_branches
        self.loss_args = training_args.loss_args
        
        # init multi loss function
        self.criterion = self.configure_loss()

    @classmethod
    def from_conf(cls, conf: DictConfig):
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        model_args = conf.model_args
        training_args = conf.training_args
        runtime_args = conf.runtime_args
        
        return cls(
            experiment_args,
            dataset_args,
            model_args,
            training_args,
            runtime_args
        )

    # Lightning framework stuff:
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        General training step
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
            "accuracy":type_acc.mean(), # accuracy computation not working
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
        # init optimizer
        optimizer = OptimizerBuilder.set_optimizer(
            self.optimizer_args, self.model
        )

        # Scheduler
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.scheduler_factor, patience=self.scheduler_patience
            ),
            'monitor': 'avg_val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def configure_loss(self):
        # Compute binary weights tensor
        binary_weights = None
        if self.class_weights:
            num_class_pixels = self.fm.get_class_pixels(self.train_data.as_posix())
            num_binary_pixels = np.sum(num_class_pixels, where=[False] + [True]*(len(num_class_pixels)-1))
            class_pixels = np.array([num_class_pixels[0], num_binary_pixels])
            binary_weights = to_device(1-class_pixels/class_pixels.sum())
        
        # compute type weight tensor
        type_weights = None
        if self.class_weights:
            class_pixels = self.fm.get_class_pixels(self.train_data.as_posix())
            type_weights = to_device(1-class_pixels/class_pixels.sum())
        
        # init loss function
        loss = LossBuilder.set_loss(
            decoder_branches_args=self.decoder_branch_args,
            loss_args=self.loss_args,
            binary_weights=binary_weights,
            type_weights=type_weights,

        )
        return loss

    def prepare_data(self):
        self.trainset = DatasetBuilder.set_train_dataset(
            decoder_branch_args=self.decoder_branch_args,
            augmentations=self.augs,
            fname=self.train_data.as_posix(), 
        )
        self.validset = DatasetBuilder.set_test_dataset(
            decoder_branch_args=self.decoder_branch_args,
            fname=self.valid_data.as_posix(), 
        )
        self.testset = DatasetBuilder.set_test_dataset(
            decoder_branch_args=self.decoder_branch_args,
            fname=self.test_data.as_posix()
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset, 
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=self.num_workers
        )
