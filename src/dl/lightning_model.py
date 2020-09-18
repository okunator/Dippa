import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from catalyst.contrib.nn import DiceLoss, IoULoss
from catalyst.contrib.nn import Ralamb, RAdam, Lookahead
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from catalyst.dl import utils
from catalyst.contrib.tools.tensorboard import SummaryItem, SummaryReader
from typing import List, Dict

from utils.file_manager import ProjectFileManager
from img_processing.augmentations import *
from datasets import *


class SegModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 dataset_args: DictConfig,
                 experiment_args: DictConfig,
                 training_args: DictConfig,
                 **kwargs
                ) -> None:
        """
        Pytorch Lightning abstraction for any pytorch segmentation model architecture used
        in this project.
        
        Args:
            model (nn.Module) : Pytorch model specification. Can be from smp, toolbelt, or a 
                                custom model. ttatch wrappers work also. Basically any model 
                                that inherits nn.Module should work
            dataset_args (DictConfig): omegaconfig DictConfig specifying arguments
                                       related to the dataset that is being used.
                                       config.py for more info
            experiment_args (DictConfig): omegaconfig DictConfig specifying arguments
                                          that are used for creating result folders and
                                          files. Check config.py for more info
            training_args (DictConfig): omegaconfig DictConfig specifying arguments
                                        that are used for training a network.
                                        Check config.py for more info
            
            
        """
        super(SegModel, self).__init__()
        
        # Hyperparams
        self.model = model
        self.batch_size = experiment_args.batch_size
        self.input_size = experiment_args.model_input_size
        self.edge_weight = training_args.loss_args.edge_weight  
        self.lr = training_args.optimizer_args.lr
        self.encoder_lr = training_args.optimizer_args.encoder_lr
        self.weight_decay = training_args.optimizer_args.weight_decay
        self.encoder_weight_decay = training_args.optimizer_args.encoder_weight_decay
        self.factor = training_args.scheduler_args.factor
        self.patience = training_args.scheduler_args.patience
        self.classes = dataset_args.classes
        self.save_hyperparameters()
        
        # Loss criterion
        self.CE = nn.CrossEntropyLoss(
            ignore_index = -100,
            reduction = "none"
        )
        
        # Filemanager
        self.fm = ProjectFileManager(
            dataset_args,
            experiment_args
        )
        
        self.n_classes = len(self.classes)
        
    
    @classmethod
    def from_conf(cls, model, conf):
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
    
        
    def __compute_metrics(self, yhat, y):
        pred = yhat.detach().cpu().numpy()
        predflat = np.argmax(pred, axis=1).flatten()
        yflat = y.cpu().numpy().flatten()
        cmatrix = confusion_matrix(yflat, predflat, labels=range(self.n_classes))
        TN = cmatrix[0, 0]
        TP = cmatrix[1, 1]
        FP = cmatrix[0, 1]
        FN = cmatrix[1, 0]
        TNR = TN / (TN + FP + 1e-08)
        TPR = TP / (TP + FN + 1e-08)
        accuracy = (TP + TN)/(TN + TP + FP + FN+ 1e-08)
        
        TNR = torch.from_numpy(np.asarray(TNR))
        TPR = torch.from_numpy(np.asarray(TPR))
        accuracy = torch.from_numpy(np.asarray(accuracy))
        
        return TNR, TPR, accuracy
    
        
    # Lightning framework stuff:
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, train_batch, batch_idx) :
        x = train_batch["image"]
        y = train_batch["mask"]
        y_weight = train_batch["mask_weight"]

        x = x.float()
        y_weight = y_weight.float()
        y = y.long()
        
        yhat = self.forward(x)
        loss_matrix = self.CE(yhat, y)
        loss = (loss_matrix * (self.edge_weight**y_weight)).mean()
        TNR, TPR, accuracy = self.__compute_metrics(yhat, y)

        logs = {
            "train_loss": loss,
            "train_accuracy":accuracy
        }
        
        return {
            "loss":loss,
            "train_accuracy":accuracy, 
            "TNR":TNR, 
            "TPR":TPR, 
            "log":logs, 
            "progress_bar": {"train_loss": loss}
        }
    
    
    def validation_step(self, val_batch, batch_idx):
        x = val_batch["image"]
        y = val_batch["mask"]
        y_weight = val_batch["mask_weight"]

        x = x.float()
        y_weight = y_weight.float()
        y = y.long()
        
        # Compute loss
        yhat = self.forward(x)
        loss_matrix = self.CE(yhat, y)        
        loss = (loss_matrix * (self.edge_weight**y_weight)).mean()
        
        # Compute confusion matrix for accuracy
        TNR, TPR, accuracy = self.__compute_metrics(yhat, y)
        
        logs = {
            "val_loss": loss, 
            "val_accuracy":accuracy
        }
        
        return {
            "val_loss": loss, 
            "val_accuracy":accuracy, 
            "TNR":TNR, 
            "TPR":TPR, 
            "log":logs, 
            "progress_bar": {"val_loss": loss}
        }
    
    
    def validation_epoch_end(self, outputs):
        accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        TNR = torch.stack([x["TNR"] for x in outputs]).mean()
        TPR = torch.stack([x["TPR"] for x in outputs]).mean()
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        tensorboard_logs = {
            "avg_val_loss": avg_val_loss, 
            "avg_val_accuracy":accuracy, 
            "avg_val_TNR":TNR, 
            "avg_val_TPR":TPR
        }
        
        return {"avg_val_accuracy":accuracy, "val_loss": avg_val_loss, "log": tensorboard_logs}
    
    
    def training_epoch_end(self, outputs):
        accuracy = torch.stack([x["train_accuracy"] for x in outputs]).mean()
        TNR = torch.stack([x["TNR"] for x in outputs]).mean()
        TPR = torch.stack([x["TPR"] for x in outputs]).mean()
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        tensorboard_logs = {
            "avg_train_loss": avg_train_loss, 
            "avg_train_accuracy":accuracy, 
            "avg_train_TNR":TNR,
            "avg_train_TPR":TPR
        }
        
        return {"avg_train_accuracy":accuracy, "avg_train_loss": avg_train_loss, "log": tensorboard_logs}

    
    def configure_optimizers(self):
        layerwise_params = {"encoder*": dict(lr=self.encoder_lr, weight_decay=self.encoder_weight_decay)}

        # Remove weight_decay for biases and apply layerwise_params for encoder
        model_params = utils.process_model_params(self.model, layerwise_params=layerwise_params)
        
        # Base Optimizer
        base_optimizer = Ralamb(
            model_params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Lookahead optimizer
        optimizer = Lookahead(base_optimizer)
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )
        return [optimizer], [scheduler]
    
    
    def prepare_data(self):
        # compose transforms
        # transforms = compose([test_transforms()])
        transforms = compose([
            hue_saturation_transforms(),
            non_rigid_transforms(),
            blur_transforms(),
            non_spatial_transforms(),
            random_crop(self.input_size),
            to_tensor()
        ])
        
        self.trainset = BinarySegmentationDataset(
            fname = self.train_data.as_posix(), 
            transforms = transforms,
        )
        
        self.testset = BinarySegmentationDataset(
            fname = self.valid_data.as_posix(), 
            transforms = transforms,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=8
        )



def plot_metrics(conf, scale: str = "log", metric: str = "loss", save:bool = False) -> None:
    """
    Plot the training and validation loss to same graph
    Args:
        scale (str) : y-axis scale. One of ("log", "normal").
        metrics (str) : One of the averaged metrics ("loss", "accuracy", "TNR", "TPR").
        save (bool): Save result image
    """
    
    assert scale in ("log", "linear"), "y-scale not in ('log', 'linear')"
    assert metric in ("loss", "accuracy", "TNR", "TPR"), "metric not in ('loss', 'accuracy', 'TNR', 'TPR')"
    ldir = Path(conf.experiment_args.experiment_root_dir)
    
    folder = f"{conf.experiment_args.model_name}/version_{conf.experiment_args.experiment_version}"
    logdir = Path(ldir / folder / "tf")
    
    train_losses_all = {}
    avg_train_losses_all = {}
    avg_valid_losses_all = {}
    avg_valid_accuracies_all = {}
    avg_train_accuracies_all = {}
    avg_valid_TNR_all = {}
    avg_train_TNR_all = {}
    avg_valid_TPR_all = {}
    avg_train_TPR_all = {}
    epochs_all = {}


    try:
        train_losses_all = []
        avg_train_losses_all = []
        avg_valid_losses_all = []
        avg_valid_accuracies_all = []
        avg_train_accuracies_all = []
        avg_valid_TNR_all = []
        avg_train_TNR_all = []
        avg_valid_TPR_all = []
        avg_train_TPR_all = []
        epochs_all = []
        summary_reader = SummaryReader(logdir, types=["scalar"])

        for item in summary_reader:
            #print(item.tag)
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
            elif item.tag == "avg_val_TNR":
                avg_valid_TNR_all.append(item.value)
            elif item.tag == "avg_train_TNR":
                avg_train_TNR_all.append(item.value)
            elif item.tag == "avg_val_TPR":
                avg_valid_TPR_all.append(item.value)
            elif item.tag == "avg_train_TPR":
                avg_train_TPR_all.append(item.value)
    except:
        pass

    np_train_losses = np.array(avg_train_losses_all)
    np_valid_losses = np.array(avg_valid_losses_all)
    np_valid_accuracy = np.array(avg_valid_accuracies_all)
    np_train_accuracy = np.array(avg_train_accuracies_all)
    np_valid_TNR = np.array(avg_valid_TNR_all)
    np_train_TNR = np.array(avg_train_TNR_all)
    np_valid_TPR = np.array(avg_valid_TPR_all)
    np_train_TPR = np.array(avg_train_TPR_all)
    np_epochs = np.unique(np.array(epochs_all))
    
    df = pd.DataFrame(
       {
            "training loss": np_train_losses, 
            "validation loss": np_valid_losses,
            "training acc":np_train_accuracy,
            "validation acc":np_valid_accuracy,
            "training TNR":np_train_TNR,
            "validation TNR":np_valid_TNR,
            "training TPR":np_train_TPR,
            "validation TPR":np_valid_TPR,
            "epoch": np_epochs[0:len(np_train_losses)]
       }
    )

    if metric == "accuracy":
        y1 = "training acc"
        y2 = "validation acc"
    elif metric == "loss":
        y1 = "training loss"
        y2 = "validation loss"
    elif metric == "TPR":
        y1 = "training TPR"
        y2 = "validation TPR"
    elif metric == "TNR":
        y1 = "training TNR"
        y2 = "validation TNR"

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
