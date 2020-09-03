import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix
from catalyst.contrib.nn import DiceLoss, IoULoss
from catalyst.contrib.nn import Ralamb, RAdam, Lookahead
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from catalyst.dl import utils
from catalyst.contrib.tools.tensorboard import SummaryItem, SummaryReader

from utils.file_manager import ProjectFileManager
from img_processing.augmentations import *
from datasets import *


class SegModel(pl.LightningModule):
    def __init__(self,
                 model,
                 dataset,
                 data_dirs,
                 database_root,
                 phases,
                 input_size,
                 batch_size,
                 edge_weight,
                 lr,
                 encoder_lr,
                 weight_decay,
                 encoder_weight_decay,
                 scheduler_factor,
                 scheduler_patience,
                 class_dict):
        """
        Pytorch Lightning abstraction for any pytorch segmentation model architecture used
        in this project.
        
        Args:
            model (nn.Module) : Pytorch model specification. Can be from smp, toolbelt, or a 
                                custom model. ttatch wrappers work also. Basically any model 
                                that inherits nn.Module should work
            dataset (str) : one of ("kumar", "consep", "pannuke", "other")
            data_dirs (dict) : dictionary of directories containing masks and images. Keys of this
                               dict must be the same as ("kumar", "consep", "pannuke", "other")
            database_root_dir (str) : directory where the databases are written
            phases (list) : list of the phases (["train", "valid", "test"] or ["train", "test"])
            input_size (int) : Size of the input patch that is fed to the network
            batch_size (int) : Number of input patches used for every iteration
            edge_weight (float) : The weight given to the borders in the cross entropy loss
            lr (float) : learning rate for the optimizer
            encoder_lr (float) : learning rate for the optimizer in the encoder part
            weight_decay (float) : weight decay for the optimizer
            encoder_weight_decay (float) : weight decay for the optimizer in the encoder part
            scheduler_factor (float) : 
            scheduler_patience (int) : ...
            class_dict (Dict) : the dict specifying pixel classes. e.g. {"background":0,"nuclei":1}
            
        """
        super(SegModel, self).__init__()
        
        # Hyperparams
        self.model = model
        self.batch_size = batch_size
        self.input_size = input_size
        self.edge_weight = edge_weight  
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.encoder_weight_decay = encoder_weight_decay
        self.factor = scheduler_factor
        self.patience = scheduler_patience
        self.classes = class_dict
        self.save_hyperparameters()
        
        # Loss criterion
        self.CE = nn.CrossEntropyLoss(
            ignore_index = -100,
            reduction = "none"
        )
        
        # Filemanager
        self.fm = ProjectFileManager(
            dataset,
            data_dirs,
            database_root,
            phases
        )
        
        self.n_classes = len(self.classes)
        
    
    @classmethod
    def from_conf(cls, model, conf):
        model = model
        dataset = conf["dataset"]["args"]["dataset"]
        data_dirs = conf["paths"]["data_dirs"]
        database_root = conf["paths"]["database_root_dir"]
        phases = conf["dataset"]["args"]["phases"]
        batch_size = conf["training_args"]["batch_size"]
        input_size = conf["patching_args"]["input_size"]
        edge_weight = conf["training_args"]["loss_args"]["edge_weight"]
        lr = conf["training_args"]["optimizer_args"]["lr"]
        encoder_lr = conf["training_args"]["optimizer_args"]["encoder_lr"]
        weight_decay = conf["training_args"]["optimizer_args"]["weight_decay"]
        encoder_weight_decay = conf["training_args"]["optimizer_args"]["encoder_weight_decay"]
        factor = conf["training_args"]["scheduler_args"]["factor"]
        patience = conf["training_args"]["scheduler_args"]["patience"]
        class_type = conf["dataset"]["args"]["class_types"]
        class_dict = conf["dataset"]["class_dicts"][class_type] # clumsy
        
        return cls(
            model,
            dataset,
            data_dirs,
            database_root,
            phases,
            input_size,
            batch_size,
            edge_weight,
            lr,
            encoder_lr,
            weight_decay,
            encoder_weight_decay,
            factor,
            patience,
            class_dict
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



def plot_metrics(conf, scale="log", metric="loss"):
    """
    Plot the training and validation loss to same graph
    Args:
        scale (str) : y-axis scale. One of ("log", "normal").
        metrics (str) : One of the averaged metrics ("loss", "accuracy", "TNR", "TPR").
    """
    
    assert scale in ("log", "linear"), "y-scale not in ('log', 'linear')"
    assert metric in ("loss", "accuracy", "TNR", "TPR"), "metric not in ('loss', 'accuracy', 'TNR', 'TPR')"
    ldir = Path(conf["paths"]["experiment_root_dir"])
    
    folder = "version_" + conf["training_args"]["experiment_version"]
    logdirs = {folder:x 
               for x in ldir.glob("**/*") 
               if x.is_dir() and str(x).split("/")[-1].startswith("tf")}

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

    for key in logdirs.keys():
        try:
            train_losses_all[key] = []
            avg_train_losses_all[key] = []
            avg_valid_losses_all[key] = []
            avg_valid_accuracies_all[key] = []
            avg_train_accuracies_all[key] = []
            avg_valid_TNR_all[key] = []
            avg_train_TNR_all[key] = []
            avg_valid_TPR_all[key] = []
            avg_train_TPR_all[key] = []
            epochs_all[key] = []
            summary_reader = SummaryReader(logdirs[key], types=["scalar"])

            for item in summary_reader:
                #print(item.tag)
                if item.tag == "train_loss":
                    train_losses_all[key].append(item.value)
                elif item.tag == "epoch":
                    epochs_all[key].append(item.value)
                elif item.tag == "avg_val_accuracy":
                    avg_valid_accuracies_all[key].append(item.value)
                elif item.tag == "avg_train_accuracy":
                    avg_train_accuracies_all[key].append(item.value)
                elif item.tag == "avg_val_loss":
                    avg_valid_losses_all[key].append(item.value)
                elif item.tag == "avg_train_loss":
                    avg_train_losses_all[key].append(item.value)
                elif item.tag == "avg_val_TNR":
                    avg_valid_TNR_all[key].append(item.value)
                elif item.tag == "avg_train_TNR":
                    avg_train_TNR_all[key].append(item.value)
                elif item.tag == "avg_val_TPR":
                    avg_valid_TPR_all[key].append(item.value)
                elif item.tag == "avg_train_TPR":
                    avg_train_TPR_all[key].append(item.value)
        except:
            pass

    np_train_losses = np.array(avg_train_losses_all[folder])
    np_valid_losses = np.array(avg_valid_losses_all[folder])
    np_valid_accuracy = np.array(avg_valid_accuracies_all[folder])
    np_train_accuracy = np.array(avg_train_accuracies_all[folder])
    np_valid_TNR = np.array(avg_valid_TNR_all[folder])
    np_train_TNR = np.array(avg_train_TNR_all[folder])
    np_valid_TPR = np.array(avg_valid_TPR_all[folder])
    np_train_TPR = np.array(avg_train_TPR_all[folder])
    np_epochs = np.unique(np.array(epochs_all[folder]))
    
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
            "epoch": np_epochs
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
    plt.show()

