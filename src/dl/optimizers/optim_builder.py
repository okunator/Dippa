
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig

import src.dl.optimizers as optims


class OptimizerBuilder:
    def __init__(self, optimizer_args: DictConfig, model: nn.Module) -> None:
        """
        Initializes the optimizer from the experiment.yml. 
        Any optimizer from torch.optim or https://github.com/jettify/pytorch-optimizer
        are allowed. 

        Args:
            optimizer_args (omegaconf.DictConfig):
                Arguments related to the optimzer
            model (nn.Module):
                pytorch model specification
        """
        self.model: nn.Module = model
        self.optimizer_name: str = optimizer_args.optimizer
        self.lr: float = optimizer_args.lr
        self.encoder_lr: float = optimizer_args.encoder_lr
        self.weight_decay: float = optimizer_args.weight_decay
        self.encoder_weight_decay: float = optimizer_args.encoder_weight_decay
        self.lookahead: bool = optimizer_args.lookahead


    def adjust_optim_params(self):
        """
        Adjust model parameters for optimizer. 

        1. Adjust learning rate and weight decay in the pre-trained encoder.
           Lower lr in encoder assumes that the encoder is already close to an optimum.
        2. Remove weight decay from bias terms to reduce overfitting

        "Bag of Tricks for Image Classification with Convolutional Neural Networks"
        https://arxiv.org/pdf/1812.01187
        """
        params = list(self.model.named_parameters())
        encoder_params = {"encoder": dict(lr=self.encoder_lr, weight_decay=self.encoder_weight_decay)}

        adjust_params = []
        for name, parameters in params:
            opts = {}
            for key, key_opts in encoder_params.items():
                if key in name:
                    for k, i in key_opts.items():
                        opts[k] = i

            if name.endswith("bias"):
                opts["weight_decay"] = 0.0
            
            adjust_params.append({"params": parameters, **opts})

        return adjust_params


    @classmethod
    def set_optimizer(cls,
                      optimizer_args: DictConfig,
                      model: nn.Module,
                      **kwargs) -> Optimizer:
        """
        Initialize the optimizer

        Args:
            optimizer_args (omegaconf.DictConfig):
                Arguments related to the optimzer
            model (nn.Module):
                pytorch model specification
        """
        c = cls(optimizer_args, model)
        optimz = list(optims.OPTIM_LOOKUP.keys())
        assert c.optimizer_name in optimz, f"optimizer: {c.optimizer_name} not one of {optimz}"

        kwargs = kwargs.copy()
        kwargs["lr"] = c.lr
        kwargs["weight_decay"] = c.weight_decay
        kwargs["params"] = c.adjust_optim_params()

        key = optims.OPTIM_LOOKUP[c.optimizer_name]
        optimizer = optims.__dict__[key](**kwargs)

        if c.lookahead:
            optimizer = optims.Lookahead(optimizer, k=5, alpha=0.5)

        return optimizer