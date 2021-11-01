import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pytorch_lightning as pl
from copy import deepcopy
from typing import List, Dict
from pathlib import Path
from omegaconf import DictConfig

from src.settings import RESULT_DIR
from src.dl.builders import LossBuilder, OptimizerBuilder, Model
from .metrics import Accuracy, MeanIoU



class SegModel(pl.LightningModule):
    def __init__(
            self,
            experiment_name: str,
            experiment_version: str,
            model_input_size: int=256,
            encoder_in_channels: int=3,
            encoder_name: str="resnet50",
            encoder_pretrain: bool=True,
            encoder_depth: int=5,
            encoder_freeze: bool=False,
            decoder_type_branch: bool=True,
            decoder_sem_branch: bool=False,
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
            sem_branch_loss: str="cd_dice",
            aux_branch_loss: str="mse_ssim",
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
            n_classes_type: int=2,
            n_classes_sem: int=2,
            class_weights: torch.Tensor=None,
            binary_weights: torch.Tensor=None,
            metrics: List[str]=["miou"],
            inference_mode: bool=False,
            **kwargs
        ) -> None:
        """
        Pytorch lightning model wrapper. Wraps everything needed for 
        training the model

        Args:
        ------------
            experiment_name (str):
                Name of the experiment. Example "Skip connection test"
            experiment_version (str):
                Name of the experiment version. Example: "dense"
            model_input_size (int, default=256):
                The input image size of the model. Assumes that input 
                images are square patches i.e. H == W. 
            encoder_name (str, default="resnet50"):
                Name of the encoder. Available encoders from:
                https://github.com/qubvel/segmentation_models.pytorch
            encoder_in_channels (int, default=3):
                Number of input channels in the encoder. Default set for
                RGB images
            encoder_pretrain (bool, default=True):
                imagenet pretrained encoder weights.
            encoder_depth (int, default=5):
                Number of encoder blocks. 
            encoder_freeze (bool, default=False):
                freeze the encoder for training
            decoder_type_branch (bool, default=True):
                Flag whether to include a cell type segmentation 
                branch to the network.
            decoder_sem_branch (bool, default=True):
                Flag whether to include a area semantic segmentation 
                branch to the network.
            decoder_aux_branch (str, default=True):
                The auxiliary branch type. One of "hover", "dist", 
                "contour", None. If None, no auxiliary branch is 
                included in the network.
            decoder_n_layers (int, default=1):
                Number of multi-conv blocks inside each level of the 
                decoder
            decoder_n_blocks (int, default=2):
                Number of conv blocks inside each multiconv block at 
                every level in the decoder.
            decoder_preactivate (bool, default=False):
                If True, normalization and activation are applied before 
                convolution
            decoder_upsampling (str, default="fixed_unpool"):
                The upsampling method. One of "interp", "max_unpool", 
                transconv", "fixed_unpool"
            decoder_weight_init (str, default="he"):
                weight initialization method One of "he", "eoc", "fixup"
                NOT IMPLEMENTED YET
            decoder_short_skips (str, default=None):
                The short skip connection style of the decoder. One of 
                ("residual", "dense", None)
            decoder_channels (List[int], default=None):
                list of integers for the number of channels in each 
                decoder block. Length of the list has to be equal to 
                encoder_depth to ensure symmetric encodedr-decoder 
                architecture.
            normalization (str): 
                Normalization method to be used.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            activation (str):
                Activation method. One of: "mish", "swish", "relu",
                "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh",
                "sigmoid", "silu", "prelu", "leaky-relu", "elu",
                "hardshrink", "tanhshrink", "hardsigmoid"
            weight_standardize (bool, default=False):
                Apply weight standardization in conv layers
            long_skips (str, default="unet"):
                The long skip connection. One of unet, unet++, unet3+.
            long_skip_merge_policy (str, default="summation"):
                merge policy of the features in long skips. One of: 
                "summation", "concatenate"
            inst_branch_loss (str, defauult="cd_dice"):
                A string specifying the loss funcs used in the binary 
                segmentation branch of the network. Loss names are 
                separated with underscores e.g. "ce_dice" One of: "ce", 
                "dice", "iou", "focal", "gmse", "mse", "sce", "tversky",
                "ssim"
            type_branch_loss (str, default="ce_dice"):
                A string specifying the loss funcs used in the cell type 
                segmentation branch of the network. Loss names are 
                separated with underscores e.g. "ce_dice" One of: "ce", 
                "dice", "iou", "focal", "gmse", "mse", "sce", "tversky",
                "ssim"
            sem_branch_loss (str, default="ce_dice"):
                A string specifying the loss funcs used in the semantic 
                segmentation branch of the network. Loss names are
                separated with underscores e.g. "ce_dice" One of: "ce", 
                "dice", "iou", "focal", "gmse", "mse", "sce", "tversky",
                "ssim"
            aux_branch_loss (str, default="mse_ssim"):
                A string specifying the loss funcs used in the auxiliary 
                regression branch of the network. Loss names are 
                separated with underscores e.g. "mse_ssim" One of: "ce",
                "gmse", "mse", "ssim"
            class_weights (bool, default=False): 
                Flag to signal wether class weights are applied in the 
                loss functions. Class weights need to be pre-computed 
                and stored in the training data dbs if this param is set
                to True
            edge_weight (float, default=None): 
                The value of the weight given at the nuclei edges. If 
                this is None, no weighting at the borders is done. Works
                only with cross-entropy based losses.
            optimizer_name (str, default="adam"):
                Name of the optimizer. In-built optimizers from torch 
                and torch_optimizer package can be used. One of: "adam", 
                "rmsprop","sgd", "adadelta", "apollo", "adabelief", 
                "adamp", "adagrad", "adamax", "adamw", "asdg", "accsgd",
                "adabound", "adamod", "diffgrad", "lamb", "novograd", 
                "pid", "qhadam", "qhm", "radam", "sgdw", "yogi", 
                "ranger","rangerqh","rangerva"
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
                List of augmentations to be used for training One of: 
                "rigid", "non_rigid", "hue_sat", "blur", "non_spatial",
                "random_crop", "center_crop","resize"
            normalize_input (bool, default=True):
                If True, channel-wise min-max normalization for the 
                input images is applied.
            batch_size (int, default=8):
                Batch size for the model at training time
            num_workers (int, default=8):
                Number of workers for the dataloader at training time
            n_classes_type (int, default=2):
                The number of cell type classes in the data.
            n_classes_sem (int, default=2):
                The number of semantic area classes in the data.
            class_weights (torch.Tensor, default=None):
                A tensor defining the weights (0 < w < 1) for the 
                different classes in the loss function
            binary_weights (torch.Tensor, default=None):
                A tensor defining the weights (0 < w < 1) for the back 
                and foreground in the loss function
            metrics (List[str], default=["miou"]):
                List of metrics that are computed during training and
                validation. Allowed metrics for now: accuracy ("acc"),
                mean-iou ("miou").
            inference_mode (bool, default=False):
                Flag to signal that model is initialized for inference. 
                This is only used in the Inferer class so no need to 
                touch this argument.
        """
        assert all(m in ("acc", "miou") for m in metrics), "Illegal metric."  
        super(SegModel, self).__init__()
        self.experiment_name = experiment_name
        self.experiment_version = experiment_version
        self.model_input_size = model_input_size
        self.n_classes_type = n_classes_type
        self.n_classes_sem = n_classes_sem
        self.metric_list = metrics

        # Encoder args
        self.encoder_in_channels = encoder_in_channels
        self.encoder_name = encoder_name
        self.encoder_pretrain = encoder_pretrain
        self.encoder_depth = encoder_depth
        self.encoder_freeze = encoder_freeze

        # Decoder_args
        self.decoder_type_branch = decoder_type_branch
        self.decoder_sem_branch = decoder_sem_branch
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
        self.sem_branch_loss = sem_branch_loss
        self.aux_branch_loss = aux_branch_loss
        self.edge_weight = edge_weight
        self.class_weights = class_weights
        self.binary_weights = binary_weights

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
        self.save_hyperparameters()

        # init model
        self.model = Model(
            encoder_name=self.encoder_name,
            encoder_in_channels=self.encoder_in_channels,
            encoder_pretrain=self.encoder_pretrain,
            encoder_depth=self.encoder_depth,
            encoder_freeze=self.encoder_freeze,
            decoder_type_branch=self.decoder_type_branch,
            decoder_aux_branch=self.decoder_aux_branch,
            decoder_sem_branch=self.decoder_sem_branch,
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
            n_classes_type=self.n_classes_type,
            n_classes_sem=self.n_classes_sem,
            model_input_size=self.model_input_size
        )

        if not inference_mode:
            # init multi loss function
            self.criterion = self.configure_loss()

            # init torchmetrics
            md = {
                "acc": {
                    "type_acc": Accuracy(),
                },
                "miou": {
                    "type_miou": MeanIoU(),
                }
            }

            if self.decoder_sem_branch:
                md["miou"]["sem_miou"] = MeanIoU()

            metrics = nn.ModuleDict(
                {k: v for m in self.metric_list for k, v in md[m].items()}
            )
            
            self.train_metrics = deepcopy(metrics)
            self.val_metrics = deepcopy(metrics)
            self.test_metrics = deepcopy(metrics)

    @classmethod
    def from_conf(cls, 
                  conf: DictConfig, 
                  **kwargs):
        """
        Construct SegModel from experiment.yml

        Args:
        ---------
            conf (omegaconf.DictConfig):
                The experiment.yml file. (File is read by omegaconf)
        """
        return cls(
            experiment_name=conf.experiment_args.experiment_name,
            experiment_version=conf.experiment_args.experiment_version,
            model_input_size=conf.training_args.input_args.model_input_size,
            encoder_in_channels=conf.model_args.architecture_design.encoder_args.in_channels,
            encoder_name=conf.model_args.architecture_design.encoder_args.encoder,
            encoder_pretrain=conf.model_args.architecture_design.encoder_args.pretrain,
            encoder_depth=conf.model_args.architecture_design.encoder_args.depth,
            encoder_freeze=conf.training_args.freeze_encoder,
            decoder_type_branch=conf.model_args.decoder_branches.type_branch,
            decoder_sem_branch=conf.model_args.decoder_branches.sem_branch,
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
            sem_branch_loss=conf.training_args.loss_args.sem_branch_loss,
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
            normalize_input=conf.training_args.input_args.normalize_input,
            batch_size=conf.runtime_args.batch_size,
            num_workers=conf.runtime_args.num_workers,
            n_classes_type=conf.training_args.input_args.n_classes_type,
            n_classes_sem=conf.training_args.input_args.n_classes_sem,
            metrics=conf.runtime_args.metrics,
            **kwargs
        )

    @classmethod
    def from_experiment(
        cls,
        name: str,
        version: str,
        inference_mode: bool=False
    ):
        """
        Construct SegModel from experiment name and version
        """
        experiment_dir = Path(f"{RESULT_DIR}/{name}/version_{version}")
        assert experiment_dir.exists(), (
            f"experiment dir: {experiment_dir} does not exist"
        )

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
        kwargs["inference_mode"] = inference_mode
        return cls(**kwargs)


    def _compute_metrics(
            self,
            preds: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            phase: str,
            key: str,
        ) -> Dict[str, torch.Tensor]:
        """
        Compute all the given torchmetrics
        """
        if phase == "train":
            metrics_dict = self.train_metrics
        elif phase == "val":
            metrics_dict = self.val_metrics
        elif phase == "test":
            metrics_dict = self.test_metrics
        
        ret = {}
        for k, metric in metrics_dict.items():
            ret[k] = metric(preds[key], targets[key], "softmax")
            if "sem" in k:
                ret[k] = metric(preds["sem"], targets["sem"], "softmax")
            
        return ret

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
            phase: str
        ) -> Dict[str, torch.Tensor]:
        """
        General training step. Runs all the required computations based 
        on the network architecture and other parameters.
        """
        # Get data
        targets = {}
        x = batch["image"].float()
        targets["instances"] = batch["binary_map"].long()

        targets["weight"] = None
        if self.edge_weight:
            targets["weight"] = batch["weight_map"].float()
            
        targets["types"] = None
        if self.decoder_type_branch:
            targets["types"] = batch["type_map"].long()

        targets["sem"] = None
        if self.decoder_sem_branch:
            targets["sem"] = batch["sem_map"].long()

        targets["aux"] = None
        if self.decoder_aux_branch is not None:
            targets["aux"] = batch["aux_map"].float()
            
        # Forward pass
        soft_masks = self.forward(x)

        # Compute loss
        loss = self.criterion(
            yhat_inst=soft_masks["instances"],
            target_inst=targets["instances"],
            yhat_type=soft_masks["types"],
            target_type=targets["types"],
            yhat_aux=soft_masks["aux"],
            target_aux=targets["aux"],
            yhat_sem=soft_masks["sem"],
            target_sem=targets["sem"],
            target_weight=targets["weight"],
            edge_weight=1.1
        )

        # Compute metrics for monitoring
        key = "types" if self.decoder_type_branch else "instances" 
        metrics = self._compute_metrics(soft_masks, targets, phase, key)

        ret = {
            "soft_masks": soft_masks if phase == "val" else None,
            "loss": loss,
        }

        return {**ret, **metrics}

    def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
        ) -> Dict[str, torch.Tensor]:
        """
        Training step + train metric logs
        """
        res = self.step(batch, batch_idx, "train")

        del res["soft_masks"] # soft masks not needed at train step
        loss = res.pop("loss")

        # log all the metrics
        self.log(
            "train_loss", loss, prog_bar=True, on_epoch=False, on_step=True
        )
        for k, val in res.items():
            self.log(
                f"train_{k}", val, prog_bar=True, on_epoch=False, on_step=True
            )

        return loss

    def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
        ) -> Dict[str, torch.Tensor]:
        """
        Validation step + validation metric logs + example outputs for
        logging
        """
        res = self.step(batch, batch_idx, "val")

        soft_masks = res.pop("soft_masks") # soft masks for logging
        loss = res.pop("loss")

        # log all the metrics
        self.log(
            "val_loss", loss, prog_bar=False, on_epoch=True, on_step=False
        )
        for k, val in res.items():
            self.log(
                f"val_{k}", val, prog_bar=False, on_epoch=True, on_step=False
            )
        
        # If batch_idx = 0, sends outputs to wandb logger
        if batch_idx in (0, 10):
            return soft_masks
        
    def test_step(
            self,
            batch: Dict[str, torch.Tensor], 
            batch_idx: int
        ) -> Dict[str, torch.Tensor]:
        """
        Test step + test metric logs
        """
        res = self.step(batch, batch_idx, "test")

        del res["soft_masks"] # soft masks not needed at test step
        loss = res.pop("loss")
        
        # log all the metrics
        self.log(
            "val_loss", loss, prog_bar=False, on_epoch=True, on_step=False
        )
        for k, val in res.items():
            self.log(
                f"val_{k}", val, prog_bar=False, on_epoch=True, on_step=False
            )

    
    def configure_optimizers(self):
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

        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                factor=self.scheduler_factor, 
                patience=self.scheduler_patience
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [scheduler]

    def configure_loss(self):
        loss = LossBuilder.set_loss(
            decoder_type_branch=self.decoder_type_branch,
            decoder_sem_branch=self.decoder_sem_branch,
            decoder_aux_branch=self.decoder_aux_branch,
            inst_branch_loss=self.inst_branch_loss,
            type_branch_loss=self.type_branch_loss,
            sem_branch_loss=self.sem_branch_loss,
            aux_branch_loss=self.aux_branch_loss,
            binary_weights=self.binary_weights,
            class_weights=self.class_weights,
            edge_weight=self.edge_weight
        )
        return loss