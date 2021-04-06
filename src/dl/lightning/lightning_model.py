import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pytorch_lightning as pl

from typing import List, Dict
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.settings import RESULT_DIR
from src.utils.file_manager import FileManager
from src.dl.datasets.dataset_builder import DatasetBuilder
from src.dl.optimizers.optim_builder import OptimizerBuilder
from src.dl.losses.loss_builder import LossBuilder
from src.dl.models.model_builder import Model
from src.dl.torch_utils import to_device
from .metrics.functional import iou


class SegModel(pl.LightningModule):
    def __init__(self,
                 experiment_name: str,
                 experiment_version: str,
                 train_dataset: str,
                 model_input_size: int=256,
                 encoder_in_channels: int=3,
                 encoder_name: str="resnet50",
                 encoder_pretrain: bool=True,
                 encoder_depth: int=5,
                 encoder_freeze: bool=False,
                 decoder_type_branch: bool=True,
                 decoder_aux_branch: str="hover",
                 decoder_n_layers: int=1,
                 decoder_n_blocks: int=2,
                 decoder_preactivate: bool=False,
                 decoder_weight_init: str="he",
                 decoder_short_skips: str=None,
                 decoder_upsampling: str="fixed_unpool",
                 decoder_channels: List[int]=None,
                 activation: str="relu",
                 normalization: str="bn",
                 weight_standardize: bool=False,
                 long_skips: str="unet",
                 long_skip_merge_policy: str="summation",
                 inst_branch_loss: str="ce_dice",
                 type_branch_loss: str="cd_dice",
                 aux_branch_loss: str="mse_ssim",
                 class_weights: bool=False,
                 edge_weight: float=None,
                 optimizer_name: str="adam",
                 decoder_learning_rate: float=0.0005,
                 encoder_learning_rate: float=0.00005, 
                 decoder_weight_decay: float=0.0003, 
                 encoder_weight_decay: float=0.00003,
                 scheduler_factor: float=0.25,
                 scheduler_patience: int=3,
                 lookahead: bool=False,
                 bias_weight_decay: bool=True,
                 augmentations: List[str]=None,
                 normalize_input: bool=True,
                 batch_size: int=8,
                 num_workers: int=8,
                 db_type: str="hdf5",
                 **kwargs) -> None:
        """
        Pytorch lightning model wrapper. Wraps everything needed for training the model

        Args:
        ------------
            experiment_name (str):
                Name of the experiment
            experiment_version (str):
                Name of the experiment version
            train_dataset (str):
                Name of thre training dataset.
                One of ("consep", "pannuke", "kumar")
            model_input_size (int, default=256):
                The input image size of the model. Assumes that input images are square
                patches i.e. H == W.
            encoder_name (str, default="resnet50"):
                Name of the encoder. Available encoders from:
                https://github.com/qubvel/segmentation_models.pytorch
            encoder_in_channels (int, default=3):
                Number of input channels in the encoder. Default set for RGB images
            encoder_pretrain (bool, default=True):
                imagenet pretrained encoder weights.
            encoder_depth (int, default=5):
                Number of encoder blocks. 
            encoder_freeze (bool, default=False):
                freeze the encoder for training
            decoder_type_branch (bool, default=True):
                Flag whether to include a type semantic segmentation branch to the network.
            decoder_aux_branch (str, default=True):
                The auxiliary branch type. One of ("hover", "dist", "contour", None). If None, no
                auxiliary branch is included in the network.
            decoder_n_layers (int, default=1):
                Number of multi-conv blocks inside each level of the decoder
            decoder_n_blocks (int, default=2):
                Number of conv blocks inside each multiconv block at every level
                in the decoder.
            decoder_preactivate (bool, default=False):
                If True, normalization and activation are applied before convolution
            decoder_upsampling (str, default="fixed_unpool"):
                The upsampling method. One of ("interp", "max_unpool", transconv", "fixed_unpool")
            decoder_weight_init (str, default="he"):
                weight initialization method One of ("he", "eoc", "fixup")
            decoder_short_skips (str, default=None):
                The short skip connection style of the decoder. One of 
                ("residual", "dense", None)
            decoder_channels (List[int], default=None):
                list of integers for the number of channels in each decoder block.
                Length of the list has to be equal to encoder_depth.
            activation (str, default="relu"):
                Activation method. One of ("mish", "swish", "relu")
            normalization (str, default="bn"):
                Normalization method. One of ("bn", "bcn" None)
            weight_standardize (bool, default=False):
                Apply weight standardization in conv layers
            long_skips (str, default="unet"):
                The long skip connection style. One of (unet, unet++, unet3+).
            long_skip_merge_policy (str, default="summation"):
                How to merge the features in long skips. One of ("summation", "concatenate")
            inst_branch_loss (str, defauult="cd_dice"):
                A string specifying the loss funcs used in the binary segmentation branch
                of the network. Loss names separated with underscores e.g. "ce_dice"
            type_branch_loss (str), default="ce_dice":
                A string specifying the loss funcs used in the semantic segmentation branch
                of the network. Loss names separated with underscores e.g. "ce_dice"
            aux_branch_loss (str, default="mse_ssim"):
                A string specifying the loss funcs used in the auxiliary regression branch
                of the network. Loss names separated with underscores e.g. "mse_ssim"
            class_weights (bool, default=False): 
                Flag to signal wheter class weights are applied in the loss functions
            edge_weight (float, default=None): 
                Weight given at the nuclei edges
            optimizer_name (str, default="adam"):
                Name of the optimizer. In-built torch optims and torch_optimizer lib 
                optimizers can be used.
            lookahead (bool, default=False):
                Flag whether the optimizer uses lookahead.
            decoder_learning_rate (float, default=0.0005):
                Decoder learning rate.
            decoder_weight_decay (float, defauilt=0.0003):
                Decoder weight decay
            encoder_weight_decay (float, default=0.00005):
                Encoder weight decay
            bias_weight_decay (bool):
                Flag whether to apply weight decay for biases.
            augmentations (List[str], default=None): 
                List of augmentations to be used for training
                One of ("rigid", "non_rigid", "hue_sat", "blur", "non_spatial",
                "random_crop", "center_crop", "resize")
            normalize_input (bool, default=True):
                If True, channel-wise normalization for the input images is applied.
            batch_size (int, default=8):
                Batch size for training
            num_workers (int, default=8):
                Number of workers for the dataloader
            db_type (str, default="hdf5"):
                Training/testing patches are saved in either hdf5 or zarr db's.
                One of ("hdf5", "zarr").  
            
        """
        super(SegModel, self).__init__()
        self.experiment_name = experiment_name
        self.experiment_version = experiment_version
        self.train_dataset = train_dataset
        self.model_input_size = model_input_size

        # Encoder args
        self.encoder_in_channels = encoder_in_channels
        self.encoder_name = encoder_name
        self.encoder_pretrain = encoder_pretrain
        self.encoder_depth = encoder_depth
        self.encoder_freeze = encoder_freeze

        # Decoder_args
        self.decoder_type_branch = decoder_type_branch
        self.decoder_aux_branch = decoder_aux_branch
        self.decoder_n_layers = decoder_n_layers
        self.decoder_n_blocks = decoder_n_blocks
        self.decoder_preactivate = decoder_preactivate
        self.decoder_weight_init = decoder_weight_init
        self.decoder_short_skips = decoder_short_skips
        self.decoder_upsampling = decoder_upsampling
        self.decoder_channels = decoder_channels
        self.long_skips = long_skips
        self.long_skip_merge_policy = long_skip_merge_policy

        # Module args
        self.activation = activation
        self.normalization = normalization
        self.weight_standardize = weight_standardize

        # Loss args
        self.inst_branch_loss = inst_branch_loss
        self.type_branch_loss = type_branch_loss
        self.aux_branch_loss = aux_branch_loss
        self.edge_weight = edge_weight
        self.class_weights = class_weights

        # Optimizer args
        self.optimizer_name = optimizer_name
        self.decoder_learning_rate = decoder_learning_rate
        self.encoder_learning_rate = encoder_learning_rate 
        self.decoder_weight_decay = decoder_weight_decay 
        self.encoder_weight_decay = encoder_weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.lookahead = lookahead
        self.bias_weight_decay = bias_weight_decay

        # Dataset & Dataloader args
        self.augmentations = augmentations
        self.normalize_input = normalize_input
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.db_type = db_type

        # save args to a file
        self.save_hyperparameters()

        # init file manager
        self.fm = FileManager(
            experiment_name=self.experiment_name,
            experiment_version=self.experiment_version
        )

        # database paths
        self.db_dict = self.fm.get_databases(self.train_dataset, db_type=self.db_type)
        self.train_data = self.db_dict['train']
        self.valid_data = self.db_dict['valid']
        self.test_data = self.db_dict['test']

        # init model
        self.model = Model(
            encoder_name=self.encoder_name,
            encoder_in_channels=self.encoder_in_channels,
            encoder_pretrain=self.encoder_pretrain,
            encoder_depth=self.encoder_depth,
            encoder_freeze=self.encoder_freeze,
            decoder_type_branch=self.decoder_type_branch,
            decoder_aux_branch=self.decoder_aux_branch,
            decoder_n_layers=self.decoder_n_layers,
            decoder_n_blocks=self.decoder_n_blocks,
            decoder_preactivate=self.decoder_preactivate,
            decoder_upsampling=self.decoder_upsampling,
            decoder_weight_init=self.decoder_weight_init,
            decoder_short_skips=self.decoder_short_skips,
            decoder_channels=self.decoder_channels,
            long_skips=self.long_skips,
            long_skip_merge_policy=self.long_skip_merge_policy,
            activation=self.activation,
            normalization=self.normalization,
            weight_standardize=self.weight_standardize,
            n_types=len(self.fm.get_classes(self.train_dataset)),
            model_input_size=self.model_input_size
        )

        # init multi loss function
        self.criterion = self.configure_loss()

        # init pl metrics
        self.accuracy = pl.metrics.Accuracy()

    @classmethod
    def from_conf(cls, conf: DictConfig):
        """
        Construct SegModel from experiment.yml
        """
        return cls(
            experiment_name=conf.experiment_args.experiment_name,
            experiment_version=conf.experiment_args.experiment_version,
            train_dataset=conf.dataset_args.train_dataset,
            model_input_size=conf.runtime_args.model_input_size,
            encoder_in_channels=conf.model_args.architecture_design.encoder_args.in_channels,
            encoder_name=conf.model_args.architecture_design.encoder_args.encoder,
            encoder_pretrain=conf.model_args.architecture_design.encoder_args.pretrain,
            encoder_depth=conf.model_args.architecture_design.encoder_args.depth,
            encoder_freeze=conf.training_args.freeze_encoder,
            decoder_type_branch=conf.model_args.decoder_branches.type_branch,
            decoder_aux_branch=conf.model_args.decoder_branches.aux_branch,
            decoder_upsampling=conf.model_args.architecture_design.decoder_args.upsampling,
            decoder_n_layers=conf.model_args.architecture_design.decoder_args.n_layers,
            decoder_n_blocks=conf.model_args.architecture_design.decoder_args.n_blocks,
            decoder_preactivate=conf.model_args.architecture_design.decoder_args.preactivate,
            decoder_weight_init=conf.model_args.architecture_design.module_args.weight_init,
            decoder_short_skips=conf.model_args.architecture_design.decoder_args.short_skips,
            decoder_channels=conf.model_args.architecture_design.decoder_args.decoder_channels,
            long_skips=conf.model_args.architecture_design.decoder_args.long_skips,
            long_skip_merge_policy=conf.model_args.architecture_design.decoder_args.merge_policy,
            activation=conf.model_args.architecture_design.module_args.activation,
            normalization=conf.model_args.architecture_design.module_args.normalization,
            weight_standardize=conf.model_args.architecture_design.module_args.weight_standardize,
            inst_branch_loss=conf.training_args.loss_args.inst_branch_loss,
            type_branch_loss=conf.training_args.loss_args.type_branch_loss,
            aux_branch_loss=conf.training_args.loss_args.aux_branch_loss,
            edge_weight=conf.training_args.loss_args.edge_weight,
            class_weights=conf.training_args.loss_args.class_weights,
            optimizer_name=conf.training_args.optimizer_args.optimizer,
            decoder_learning_rate=conf.training_args.optimizer_args.lr,
            encoder_learning_rate=conf.training_args.optimizer_args.encoder_lr, 
            decoder_weight_decay=conf.training_args.optimizer_args.weight_decay, 
            encoder_weight_decay=conf.training_args.optimizer_args.encoder_weight_decay,
            scheduler_factor=conf.training_args.optimizer_args.scheduler_factor,
            scheduler_patience=conf.training_args.optimizer_args.scheduler_patience,
            lookahead=conf.training_args.optimizer_args.lookahead,
            bias_weight_decay=conf.training_args.optimizer_args.bias_weight_decay,
            augmentations=conf.training_args.augmentations,
            normalize_input=conf.training_args.normalize_input,
            batch_size=conf.runtime_args.batch_size,
            num_workers=conf.runtime_args.num_workers,
            db_type=conf.runtime_args.db_type,
        )

    @classmethod
    def from_experiment(cls, name: str, version: str):
        """
        Construct SegModel from experiment name and version
        """
        experiment_dir = Path(f"{RESULT_DIR}/{name}/version_{version}")
        assert experiment_dir.exists(), f"experiment dir: {experiment_dir} does not exist"

        # open meta_tags.csv
        for obj in experiment_dir.iterdir():
            if obj.name == "meta_tags.csv":
                df = pd.read_csv(obj)

        # set kwargs
        records = df.to_dict("records")
        kwargs = dict((d["key"], d["value"]) for d in records)

        # Convert lists, bools, and ints from str
        for k, v in kwargs.items():
            try:
                kwargs[k] = eval(v)
            except:
                pass
            
            try:
                kwargs[k] = int(v)
            except:
                pass

            if kwargs[k] == "TRUE" or kwargs[k] == "FALSE":
                kwargs[k] = kwargs[k] == "TRUE"

        kwargs["experiment_name"] = name
        kwargs["experiment_version"] = version
        return cls(**kwargs)

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
        if self.decoder_type_branch:
            type_target = batch["type_map"].long()

        aux_target = None
        if self.decoder_aux_branch is not None:
            if self.decoder_aux_branch == "hover":
                xmap = batch["xmap"].float()
                ymap = batch["ymap"].float()
                aux_target = torch.stack([xmap, ymap], dim=1)
            elif self.decoder_aux_branch == "dist":
                aux_target = batch["dist_map"].float()
                aux_target = aux_target.unsqueeze(dim=1)
            elif self.decoder_aux_branch == "contour":
                aux_target = batch["contour"].float()
                aux_target = aux_target.unsqueeze(dim=1)

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
        key = "types" if self.decoder_type_branch else "instances" 
        type_acc = self.accuracy(soft_mask[key], type_target)
        type_iou = iou(soft_mask[key], type_target, "softmax")

        return {
            "loss":loss,
            "accuracy":type_acc, # accuracy computation not working
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

        prog_bar = phase == "train"
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=prog_bar, logger=True)

        return {
            "loss": z["loss"],
            "accuracy": z["accuracy"],
            "mean_iou": z["mean_iou"]
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

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return {f"avg_{phase}_loss": loss}
    
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
    
    def training_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end(outputs, "train")
        
    def validation_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end(outputs, "test")

    def configure_optimizers(self):
        # init optimizer
        optimizer = OptimizerBuilder.set_optimizer(
            optimizer_name=self.optimizer_name,
            lookahead=self.lookahead,
            model=self.model,
            decoder_learning_rate=self.decoder_learning_rate,
            encoder_learning_rate=self.encoder_learning_rate,
            decoder_weight_decay=self.decoder_weight_decay,
            encoder_weight_decay=self.encoder_weight_decay,
            bias_weight_decay=self.bias_weight_decay
        )

        # Scheduler
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.scheduler_factor, patience=self.scheduler_patience
            ),
            "monitor": "avg_val_loss",
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [scheduler]

    def configure_loss(self):
        # Compute binary weights tensor
        self.binary_weights = None
        if self.class_weights:
            weights = self.fm.get_class_weights(self.train_data.as_posix(), binary=True)
            self.binary_weights = to_device(weights)
        
        # compute type weight tensor
        self.type_weights = None
        if self.class_weights:
            weights = self.fm.get_class_weights(self.train_data.as_posix())
            self.type_weights = to_device(weights)
        
        # init loss function
        loss = LossBuilder.set_loss(
            decoder_type_branch=self.decoder_type_branch,
            decoder_aux_branch=self.decoder_aux_branch,
            inst_branch_loss=self.inst_branch_loss,
            type_branch_loss=self.type_branch_loss,
            aux_branch_loss=self.aux_branch_loss,
            binary_weights=self.binary_weights,
            class_weights=self.type_weights,
            edge_weight=self.edge_weight
        )
        return loss

    def prepare_data(self):
        self.trainset = DatasetBuilder.set_train_dataset(
            fname=self.train_data.as_posix(),
            augmentations=self.augmentations,
            decoder_aux_branch=self.decoder_aux_branch,
            normalize_input=self.normalize_input
        )
        self.validset = DatasetBuilder.set_test_dataset(
            fname=self.valid_data.as_posix(),
            decoder_aux_branch=self.decoder_aux_branch,
            normalize_input=self.normalize_input
        )
        self.testset = DatasetBuilder.set_test_dataset(
            fname=self.test_data.as_posix(),
            decoder_aux_branch=self.decoder_aux_branch,
            normalize_input=self.normalize_input
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=self.num_workers,
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
