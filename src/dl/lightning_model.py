import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from catalyst.contrib.nn import Ralamb, RAdam, Lookahead
from catalyst.dl import utils
from catalyst.contrib.tools.tensorboard import SummaryItem, SummaryReader

from src.utils.file_manager import ProjectFileManager
from src.dl.datasets.dataset_builder import DatasetBuilder
from src.dl.losses.loss_builder import LossBuilder
from src.dl.torch_utils import to_device, argmax_and_flatten, mean_iou


class SegModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 training_args: DictConfig,
                 **kwargs) -> None:
        """
        Pytorch Lightning abstraction for any pytorch segmentation model architecture used
        in this project.
        
        Args:
            model (nn.Module): Pytorch model specification. Basically any model that inherits
                nn.Module should work
            dataset_args (DictConfig): omegaconfig DictConfig specifying arguments related 
                to the dataset that is being used. config.py for more info
            experiment_args (DictConfig): omegaconfig DictConfig specifying arguments that
                are used for creating result folders and files. Check config.py for more info
            training_args (DictConfig): omegaconfig DictConfig specifying arguments that are
                used for training a network. Check config.py for more info
        """
        super(SegModel, self).__init__()
        
        # Hyperparams
        self.model = model
        self.batch_size = training_args["batch_size"]
        self.input_size = training_args["model_input_size"]
        self.loss_name_inst = training_args["inst_branch_loss"]
        self.loss_name_type = training_args["semantic_branch_loss"]
        self.loss_name_aux = training_args["aux_branch_loss"]
        self.edge_weight = training_args["edge_weight"]
        self.class_weights = training_args["class_weights"]
        self.lr = training_args["lr"]
        self.encoder_lr = training_args["encoder_lr"]
        self.weight_decay = training_args["weight_decay"]
        self.encoder_weight_decay = training_args["encoder_weight_decay"]
        self.factor = training_args["factor"]
        self.patience = training_args["patience"]
        self.save_hyperparameters()

        # Filemanager
        self.fm = ProjectFileManager(
            dataset_args,
            experiment_args
        )
        
        self.criterion = self.prepare_criterion
        # torch.autograd.set_detect_anomaly(True)
        
    @classmethod
    def from_conf(cls, model: nn.Module, conf: DictConfig):
        model = model
        dataset_args = conf.dataset_args
        experiment_args = conf.experiment_args
        training_args = conf.training_args
        
        return cls(
            model,
            dataset_args,
            experiment_args,
            training_args
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
    def step(self):
        return self.step_singlebranch if self.fm.class_types == "instance" else self.step_twobranch

    @property
    def binary_class_weights(self):
        weights = None
        if self.class_weights:
            num_class_pixels = self.fm.get_class_pixels_num(self.train_data.as_posix())
            num_binary_pixels = np.sum(num_class_pixels, where=[False] + [True]*(len(num_class_pixels)-1))
            class_pixels = np.array([num_class_pixels[0], num_binary_pixels])
            weights = to_device(1-class_pixels/class_pixels.sum())
        return weights

    @property
    def type_class_weights(self):
        weights = None
        if self.class_weights:
            class_pixels = self.fm.get_class_pixels_num(self.train_data.as_posix())
            weights = to_device(1-class_pixels/class_pixels.sum())
        return weights

    @property
    def prepare_criterion(self):
        return LossBuilder.set_loss(
            loss_name_inst=self.loss_name_inst,
            loss_name_type=self.loss_name_type,
            loss_name_aux=self.loss_name_aux,
            class_types=self.fm.class_types,
            binary_weights=self.binary_class_weights,
            type_weights=self.type_class_weights,
            edge_weight=self.edge_weight,
            aux_branch_name=self.fm.aux_branch
        )
                
    # Lightning framework stuff:
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    
    def step_singlebranch(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        For models with only one segmentation branch for instance segmentation.
        """
        x = batch["image"]
        target = batch["binary_map"]
        target_weight = batch["weight_map"]

        x = x.float()
        target_weight = target_weight.float()
        target = target.long()

        if self.fm.aux_branch == "hover":
            xmap = batch["xmap"].float() 
            ymap = batch["ymap"].float()
            target_aux = torch.stack([xmap, ymap], dim=1)
        elif self.fm.aux_branch == "micro":
            pass
        else:
            target_aux = None

        soft_mask = self.forward(x)
        loss = self.criterion(
            yhat_inst=soft_mask["instances"], 
            target_inst=target,
            target_weight=target_weight,
            yhat_aux=soft_mask["aux"],
            target_aux=target_aux
        )

        accuracy = utils.metrics.accuracy(
            argmax_and_flatten(soft_mask["instances"], "sigmoid"), target.view(1, -1)
        )

        iou = mean_iou(soft_mask["instances"], target, "sigmoid")

        return {
            "loss":loss,
            "accuracy":accuracy[0],
            "mean_iou":iou.mean()
        }

    def step_twobranch(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        For models with two decoder branches, one for inst segmentation, one for semantic. 
        """
        x = batch["image"]
        inst_target = batch["binary_map"]
        type_target = batch["type_map"]
        target_weight = batch["weight_map"]

        x = x.float()
        target_weight = target_weight.float()
        inst_target = inst_target.long()
        type_target = type_target.long()

        if self.fm.aux_branch == "hover":
            xmap = batch["xmap"].float()
            ymap = batch["ymap"].float()
            target_aux = torch.stack([xmap, ymap], dim=1)
        elif self.fm.aux_branch == "micro":
            pass
        else:
            target_aux = None
        
        soft_mask = self.forward(x)
        
        loss = self.criterion(
            yhat_inst=soft_mask["instances"], 
            yhat_type=soft_mask["types"], 
            target_inst=inst_target, 
            target_type=type_target, 
            target_weight=target_weight,
            yhat_aux=soft_mask["aux"],
            target_aux=target_aux,
        )

        type_acc = utils.metrics.accuracy(
            argmax_and_flatten(soft_mask["instances"], "softmax"), inst_target.view(1, -1)
        )

        type_iou = mean_iou(soft_mask["types"], type_target, "softmax")

        return {
            "loss":loss,
            "accuracy":type_acc[0],
            "mean_iou":type_iou.mean()
        }

    def step_return_dict(self, z: torch.Tensor, phase: str) -> Dict[str, torch.Tensor]:
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
        layerwise_params = {"encoder*": dict(lr=self.encoder_lr, weight_decay=self.encoder_weight_decay)}

        # Remove weight_decay for biases and apply layerwise_params for encoder
        model_params = utils.process_model_params(self.model, layerwise_params=layerwise_params)
        
        # Base Optimizer
        base_optimizer = Ralamb(model_params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Lookahead optimizer
        optimizer = [Lookahead(base_optimizer)]
        
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


    def plot_metrics(self, scale: str = "log", metric: str = "loss", save:bool = False) -> None:
        """
        Plot the training and validation loss to same graph

        Args:
            conf (DictConfig): the config 
            scale (str): y-axis scale. One of ("log", "normal").
            metrics (str): One of the averaged metrics ("loss", "accuracy", "TNR", "TPR").
            save (bool): Save result image
        """
        # cludge
        assert scale in ("log", "linear"), "y-scale not in ('log', 'linear')"
        assert metric in ("loss", "accuracy", "mean_iou"), "metric not in ('loss', 'accuracy', 'mean_iou')"
        folder = f"{self.fm.experiment_dir}"
        logdir = Path(f"{folder}/tf")
        train_losses_all = {}
        avg_train_losses_all = {}
        avg_valid_losses_all = {}
        avg_valid_accuracies_all = {}
        avg_train_accuracies_all = {}
        avg_valid_iou_all = {}
        avg_train_iou_all = {}
        epochs_all = {}

        try:
            train_losses_all = []
            avg_train_losses_all = []
            avg_valid_losses_all = []
            avg_valid_accuracies_all = []
            avg_train_accuracies_all = []
            avg_valid_iou_all = []
            avg_train_iou_all = []
            epochs_all = []
            summary_reader = SummaryReader(logdir, types=["scalar"])
            for item in summary_reader:
                # print(item.tag)
                if item.tag == "train_loss":
                    train_losses_all.append(item.value)
                elif item.tag == "epoch":
                    epochs_all.append(item.value)
                elif item.tag == "avg_val_accuracy":
                    avg_valid_accuracies_all.append(item.value)
                elif item.tag == "avg_train_accuracy":
                    avg_train_accuracies_all.append(item.value)
                elif item.tag == "avg_val_loss":
                    avg_valid_losses_all.append(item.value)
                elif item.tag == "avg_train_loss":
                    avg_train_losses_all.append(item.value)
                elif item.tag == "avg_val_iou":
                    avg_valid_iou_all.append(item.value)
                elif item.tag == "avg_train_iou":
                    avg_train_iou_all.append(item.value)
        except:
            pass

        np_train_losses = np.array(avg_train_losses_all)
        np_valid_losses = np.array(avg_valid_losses_all)
        np_valid_accuracy = np.array(avg_valid_accuracies_all)
        np_train_accuracy = np.array(avg_train_accuracies_all)
        np_valid_iou = np.array(avg_valid_iou_all)
        np_train_iou = np.array(avg_train_iou_all)
        np_epochs = np.unique(np.array(epochs_all))
        
        df = pd.DataFrame({
            "training loss": np_train_losses, 
            "validation loss": np_valid_losses,
            "training acc":np_train_accuracy,
            "validation acc":np_valid_accuracy,
            "training iou":np_train_iou,
            "validation iou":np_valid_iou,
            "epoch": np_epochs[0:len(np_train_losses)]
        })

        if metric == "accuracy":
            y1 = "training acc"
            y2 = "validation acc"
        elif metric == "loss":
            y1 = "training loss"
            y2 = "validation loss"
        elif metric == "mean_iou":
            y1 = "training iou"
            y2 = "validation iou"

        plt.rcParams["figure.figsize"] = (20,10)
        ax = plt.gca()
        df.plot(kind="line",x="epoch", y=y1, ax=ax)
        df.plot(kind="line",x="epoch", y=y2, color="red", ax=ax)
        plt.yscale(scale)
        
        if save:
            plot_dir = logdir.parents[0].joinpath("training_plots")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(plot_dir / f"{scale}_{metric}.png"))
        
        plt.show()
