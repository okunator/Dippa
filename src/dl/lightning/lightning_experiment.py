import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from copy import deepcopy
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf

from src.settings import RESULT_DIR
from src.dl.models import MultiTaskSegModel
from src.dl.losses.utils import multitaskloss_func
from src.dl.optimizers.utils import OptimizerBuilder

from .metrics.utils import metric_func


class SegExperiment(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            experiment_name: str,
            experiment_version: str,
            branch_losses: Dict[str, int]=None,
            metrics: Dict[str, List[str]]=None,
            optimizer_name: str="adam",
            edge_weights: Dict[str, float]=None,
            class_weights: Dict[str, bool]=None,
            dec_learning_rate: float=0.0005,
            enc_learning_rate: float=0.00005, 
            dec_weight_decay: float=0.0003, 
            enc_weight_decay: float=0.00003,
            scheduler_factor: float=0.25,
            scheduler_patience: int=3,
            lookahead: bool=False,
            bias_weight_decay: bool=True,
            dataset_type: str="hover",
            batch_size: int=8,
            normalize_input: bool=False,
            inference_mode: bool=False,
            hparams_to_yaml: bool=True,
            **kwargs
        ) -> None:
        """
        Pytorch lightning model wrapper. Wraps everything needed for 
        training the model

        Args:
        ------------
            model (nn.Module):
                A pytorch model specification
            experiment_name (str):
                Name of the experiment.
            experiment_version (str):
                Name of the experiment version.
            branch_losses (Dict[str, int], default=None)
                dictionary mapping multi-loss functions to the decoder
                branches. Allowed losses: "ce", "dice", "iou", "focal",
                "gmse", "mse", "sce", "tversky", "ssim"
            metrics (Dict[str, List[str]], default=None):
                Dictionary of branch name mapped to a List of metrics
                The metrics are computed during training and
                validation. Allowed metrics for now: accuracy ("acc"),
                mean-iou ("miou").
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
            dec_learning_rate (float, default=0.0005):
                Decoder learning rate.
            dec_weight_decay (float, defauilt=0.0003):
                Decoder weight decay
            enc_weight_decay (float, default=0.00005):
                Encoder weight decay
            bias_weight_decay (bool):
                Flag whether to apply weight decay for biases.
            dataset_type (str, default="hover"):
                The dataset type that is used to train the model.
                Specifies the auxilliary branch type if there is one.
                One of: "hover", "dist", "contour", "basic", "unet"
            batch_size (int, default=8):
                Batch size for the dataloader during training
            normalize_input (bool, default=False):
                If True, channel-wise min-max normalization is applied 
                to input imgs in the dataloading process during training
            inference_mode (bool, default=False):
                Flag to signal that model is initialized for inference. 
                This is only used in the Inferer class so no need to 
                touch this argument.
            hparams_to_yaml (bool, default=True):
                If True, all the config params are saved to a .yml file
                in the RESULT_DIR (Dippa/results/{exp_name}/{exp_vers})
        """
        super().__init__()
        
        # Save hparms   
        self.name = experiment_name
        self.version = experiment_version
        
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
        self.dataset_type = dataset_type
        self.batch_size = batch_size,
        self.normalize_input = normalize_input,

        self.save_hyperparameters(
            "experiment_name",
            "experiment_version",
            "branch_losses",
            "metrics",
            "optimizer_name",
            "edge_weights",
            "class_weights",
            "dec_learning_rate",
            "enc_learning_rate",
            "dec_weight_decay",
            "enc_weight_decay",
            "scheduler_factor",
            "scheduler_patience",
            "lookahead",
            "bias_weight_decay",
            "dataset_type",
            "batch_size",
            "normalize_input"
        )
        
        # init model
        self.model = model
        # self.save_hyperparameters(ignore=["model"])
        
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
        self.hparams["dec_attention"] = model.dec_attention
        

        if not inference_mode:
            self.metric_dict = metrics
            self.branch_losses = branch_losses
            self.edge_weights = edge_weights
            self.class_weights = class_weights
            self._validate_branch_args()
            
            self.criterion = self.configure_loss()
            metrics = self.configure_metrics()
            self.train_metrics = deepcopy(metrics)
            self.val_metrics = deepcopy(metrics)
            self.test_metrics = deepcopy(metrics)
            
        if hparams_to_yaml:
            self._hparams_to_yaml()
            
    def _validate_branch_args(self) -> None:
        """
        Check that there are no conflicting decoder branch args
        """
        lk = self.branch_losses.keys()
        dk = self.model.dec_branches.keys()
        mk = self.metric_dict.keys()
        has_same_keys = (lk == dk == mk)
              
        ek = None
        if isinstance(self.edge_weights, dict):
            ek = self.edge_weights.keys()
            has_same_keys = has_same_keys == ek

        ck = None
        if isinstance(self.class_weights, dict):
            ck = self.class_weights.keys()
            has_same_keys = has_same_keys == ck
            
        if not has_same_keys:
            raise ValueError(f"""
                Got mismatching keys for branch dict args. Branch losses: {lk}.
                Decoder branches: {dk}. Metrics: {mk}. Edge weights {ek}.
                Class weights: {ck}. Edge weights and class weights can be
                None"""
            )

    @classmethod
    def from_conf(
            cls,
            model: nn.Module,
            conf: DictConfig,
            **kwargs
        ):
        """
        Construct SegModel from experiment.yml

        Args:
        ---------
            model (nn.Module):
                A pytorch model specification
            conf (omegaconf.DictConfig):
                A config .yml file specifying metrics, optimizers,
                losses etc. (File is read by omegaconf)
        """
        train_kwds = conf.training
        data_kwds = conf.datamodule
        
        return cls(
            model=model,
            experiment_name=conf.experiment_name,
            experiment_version=conf.experiment_version,
            edge_weights=train_kwds.edge_weights,
            class_weights=train_kwds.class_weights,
            branch_losses=train_kwds.loss.branch_losses,
            optimizer_name=train_kwds.optimizer.name,
            lookahed=train_kwds.optimizer.lookahead,
            dec_learning_rate=train_kwds.optimizer.decoder_lr,
            dec_weight_decay=train_kwds.optimizer.decoder_weight_decay,
            enc_learning_rate=train_kwds.optimizer.encoder_lr,
            enc_weight_decay=train_kwds.optimizer.encoder_weight_decay,
            bias_weight_decay=train_kwds.optimizer.bias_weight_decay,
            metrics=train_kwds.metrics,
            batch_size=data_kwds.batch_size,
            dataset_type=data_kwds.dataset_type,
            normalize_input=data_kwds.normalize_input,
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
        Construct SegExperiment from experiment name and version
        """
        exp_dir = RESULT_DIR / name / f"version_{version}"
        if not exp_dir.exists():
            raise ValueError(f"experiment dir: {exp_dir} does not exist")
        
        hparams = exp_dir / "hparams_all.yml"
        conf = OmegaConf.load(hparams)
        
        model = MultiTaskSegModel(**conf)

        return cls(
            model=model, 
            inference_mode=inference_mode,
            hparams_to_yaml=False,
            **conf
        )
    
    def _hparams_to_yaml(self) -> None:
        """
        Save hparams to yaml
        """
        f = RESULT_DIR / self.name / f"version_{self.version}" / "hparams_all.yml"
        f.parents[0].mkdir(exist_ok=True, parents=True)
        conf = OmegaConf.create(dict(self.hparams.items()))
        OmegaConf.save(conf, f=f)

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
                act = None if "aux" in branch else "softmax"
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
            "train_loss", loss, prog_bar=True,
            on_epoch=False, on_step=True
        )
        for k, val in res.items():
            self.log(
                f"train_{k}", val, prog_bar=True,
                on_epoch=False, on_step=True
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
            "val_loss", loss, prog_bar=False,
            on_epoch=True, on_step=False
        )
        for k, val in res.items():
            self.log(
                f"val_{k}", val, prog_bar=False,
                on_epoch=True, on_step=False
            )
        
        # If batch_idx = 0, sends outputs to wandb logger
        if batch_idx in (0, 1, 2):
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
            "val_loss", loss, prog_bar=False,
            on_epoch=True, on_step=False
        )
        for k, val in res.items():
            self.log(
                f"val_{k}", val, prog_bar=False,
                on_epoch=True, on_step=False
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
            self.branch_losses,
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