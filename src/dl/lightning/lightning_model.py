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
from src.dl.losses.utils import multitaskloss_func
from src.dl.optimizers.utils import OptimizerBuilder

from .metrics.utils import metric_func
from ..models import MultiTaskSegModel


class SegModel(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            experiment_name: str,
            experiment_version: str,
            optimizer_name: str="adam",
            edge_weights: Dict[str, float]=None,
            class_weights: Dict[str, bool]=None,
            dec_branch_losses: Dict[str, int]={"inst": "ce_dice"},
            dec_learning_rate: float=0.0005,
            enc_learning_rate: float=0.00005, 
            dec_weight_decay: float=0.0003, 
            enc_weight_decay: float=0.00003,
            scheduler_factor: float=0.25,
            scheduler_patience: int=3,
            lookahead: bool=False,
            bias_weight_decay: bool=True,
            augmentations: List[str]=None,
            normalize_input: bool=True,
            rm_overlaps: bool=False,
            batch_size: int=8,
            metrics: Dict[str, List[str]]={"inst": ["miou"]},
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
            edge_weights (Dict[str, float], optional, default=None):
                A dictionary of baranch names mapped to floats that are used
                to weight nuclei edges in CE-based losses. e.g.
                {"inst": 1.1, "aux": None}
            class_weights (Dict[str, bool], optional, default=None):
                A dictionary of branch names mapped to a boolean.
                If boolean == True, class weights are used in the losses
                for that branch.
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
            dec_branch_losses (Dict[str, int], default={"inst": "ce_dice"})
                dictionary mapping multi-loss functions to the decoder
                branches. Allowed losses: "ce", "dice", "iou", "focal",
                "gmse", "mse", "sce", "tversky", "ssim"
            dec_learning_rate (float, default=0.0005):
                Decoder learning rate.
            dec_weight_decay (float, defauilt=0.0003):
                Decoder weight decay
            enc_weight_decay (float, default=0.00005):
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
            rm_overlaps (bool, default=False):
                If True, then the object borders are removed from the
                target masks while training.
            batch_size (int, default=8):
                Batch size for the model at training time
            metrics (Dict[str, List[str]], default={"inst":["miou"]}):
                Dictionary of branch name mapped to a List of metrics
                The metrics are computed during training and
                validation. Allowed metrics for now: accuracy ("acc"),
                mean-iou ("miou").
            inference_mode (bool, default=False):
                Flag to signal that model is initialized for inference. 
                This is only used in the Inferer class so no need to 
                touch this argument.
        """
        super().__init__()
        
        # check that dict args have matching keys
        lk = dec_branch_losses.keys()
        dk = model.dec_branches.keys()
        mk = metrics.keys()
        has_same_keys = (lk == dk == mk)
              
        ek = None
        if isinstance(edge_weights, dict):
            ek = edge_weights.keys()
            has_same_keys = has_same_keys == ek

        ck = None
        if isinstance(class_weights, dict):
            ck = class_weights.keys()
            has_same_keys = has_same_keys == ck
            
        if not has_same_keys:
            raise ValueError(f"""
                Got mismatching keys for dict args. Branch losses: {lk}.
                Decoder branches: {dk}. Metrics: {mk}. Edge weights {ek}.
                Class weights: {ck}. Edge weights and class weights can be
                None"""
            )
             
        # Save hparms   
        self.experiment_name = experiment_name
        self.experiment_version = experiment_version
        self.metric_dict = metrics
        
        # Loss args
        self.decoder_branch_losses = dec_branch_losses
        self.edge_weights = edge_weights
        self.class_weights = class_weights

        # Optimizer args
        self.optimizer_name = optimizer_name
        self.decoder_learning_rate = dec_learning_rate
        self.encoder_learning_rate = enc_learning_rate 
        self.decoder_weight_decay = dec_weight_decay 
        self.encoder_weight_decay = enc_weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.lookahead = lookahead
        self.bias_weight_decay = bias_weight_decay

        # Dataset & Dataloader args
        self.augmentations = augmentations
        self.normalize_input = normalize_input
        self.batch_size = batch_size
        self.rm_overlaps = rm_overlaps
        
        # init model
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        
        # add model hparams
        self.hparams["model_input_size"] = model.model_input_size
        self.hparams["enc_name"] = model.enc_name
        self.hparams["enc_pretrain"] = model.enc_pretrain
        self.hparams["enc_depth"] = model.enc_depth
        self.hparams["enc_freeze"] = model.enc_freeze

        self.hparams["dec_branches"] = model.dec_branches
        self.hparams["dec_n_layers"] = model.dec_n_layers
        self.hparams["dec_conv_types"] = model.dec_conv_types
        self.hparams["dec_n_blocks"] = model.dec_n_blocks
        self.hparams["dec_preactivate"] = model.dec_preactivate
        self.hparams["dec_short_skip"] = model.dec_short_skip
        self.hparams["dec_upsampling"] = model.dec_upsampling
        self.hparams["dec_channels"] = model.dec_channels
        self.hparams["dec_long_skips"] = model.dec_long_skip
        self.hparams["dec_long_skip_merge_policy"] = model.dec_long_skip_merge
        self.hparams["dec_activation"] = model.dec_activation
        self.hparams["dec_normalization"] = model.dec_normalization
        self.hparams["dec_weight_standardize"] = model.dec_weight_standardize

        if not inference_mode:
            self.criterion = self.configure_loss()
            metrics = self.configure_metrics()
            self.train_metrics = deepcopy(metrics)
            self.val_metrics = deepcopy(metrics)
            self.test_metrics = deepcopy(metrics)

    @classmethod
    def from_conf(cls, conf: DictConfig, **kwargs):
        """
        Construct SegModel from experiment.yml

        Args:
        ---------
            conf (omegaconf.DictConfig):
                The experiment.yml file. (File is read by omegaconf)
        """
        model = MultiTaskSegModel.from_conf(conf)
        train_kwds = conf.training
        
        return cls(
            model=model,
            experiment_name=conf.expriment_name,
            experiment_version=conf.experiment_version,
            edge_weights=train_kwds.edge_weights,
            class_weights=train_kwds.class_weights,
            dec_branch_losses=train_kwds.loss.branch_losses,
            optimizer_name=train_kwds.optimizer.name,
            lookahed=train_kwds.optimizer.lookahead,
            dec_learning_rate=train_kwds.optimizer.decoder_lr,
            dec_weight_decay=train_kwds.optimizer.decoder_weight_decay,
            enc_learning_rate=train_kwds.optimizer.encoder_lr,
            enc_weight_decay=train_kwds.optimizer.encoder_weight_decay,
            bias_weight_decay=train_kwds.optimizer.bias_weight_decay,
            augmentations=train_kwds.input.augmentations,
            normalize_input=train_kwds.input.normalize_input,
            rm_overlaps=train_kwds.input.rm_overlaps,
            num_workers=train_kwds.num_workers,
            batch_size=train_kwds.batch_size,
            metrics=train_kwds.metrics,
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
            branch = k.split("_")[0]
            if metric is not None:
                act = None if branch == "aux" else "softmax"
                ret[k] = metric(
                    preds[f"{branch}_map"], targets[f"{branch}_map"], act
                )
            
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
        x = batch["image"].float()
        
        # Forward pass
        soft_masks = self.forward(x)
        branch_maps = soft_masks.keys()
        
        # Targets for the loss
        targets = {}
        
        targets["inst_map"] = None
        if "inst_map" in branch_maps:
            targets["inst_map"] = batch["binary_map"].long()

        targets["type_map"] = None
        if "type_map" in branch_maps:
            targets["type_map"] = batch["type_map"].long()

        targets["sem_map"] = None
        if "sem_map" in branch_maps:
            targets["sem_map"] = batch["sem_map"].long()

        targets["aux_map"] = None
        if "aux_map" in branch_maps:
            targets["aux_map"] = batch["aux_map"].float()
            
        targets["weight"] = None
        if self.edge_weights:
            targets["weight_map"] = batch["weight_map"].float()
        
        # Compute loss
        loss = self.criterion(yhats=soft_masks, targets=targets)

        # Compute metrics for monitoring
        metrics = self._compute_metrics(soft_masks, targets, phase)

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
        loss = multitaskloss_func(
            self.decoder_branch_losses,
            self.edge_weights,
            # self.class_weights TODO: fix this
        )
        
        return loss
    
    def configure_metrics(self):        
        metrics = nn.ModuleDict()
        for k, m in self.metric_dict.items():
            for metric in m:
                metrics[f"{k}_{metric}"] = metric_func(metric)
            
        return metrics