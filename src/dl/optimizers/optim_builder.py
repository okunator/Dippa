
import torch.nn as nn
from torch.optim.optimizer import Optimizer

import src.dl.optimizers as optims


class OptimizerBuilder:
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float,
                 encoder_learning_rate: float,
                 weight_decay: float,
                 encoder_weight_decay: float,
                 bias_weight_decay: bool) -> None:
        """
        Class used to initialize the optimizer from the given args
        Any optimizer from torch.optim or https://github.com/jettify/pytorch-optimizer
        are allowed.

        Args:
        -----------
            model (nn.Module):
                pytorch model specification
            optimizer_name (str):
                Name of the optimize. In-built torch optims and torch_optimizer lib 
                optimizers can be used.
            learning_rate (float):
                Decoder learning rate.
            weight_decay (float):
                decoder weight decay
            encoder_weight_decay (float):
                encoder weight decay
            bias_weight_decay (bool):
                Flag whether to apply weight decay for biases.
        """
        self.model: nn.Module = model
        self.lr = learning_rate
        self.encoder_lr = encoder_learning_rate
        self.weight_decay = weight_decay
        self.encoder_weight_decay = encoder_weight_decay
        self.bias_weight_decay = bias_weight_decay


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
                        
            if self.bias_weight_decay:
                if name.endswith("bias"):
                    opts["weight_decay"] = 0.0
            
            adjust_params.append({"params": parameters, **opts})

        return adjust_params


    @classmethod
    def set_optimizer(cls,
                      optimizer_name: str,
                      lookahead: bool,
                      model: nn.Module,
                      learning_rate: float,
                      encoder_learning_rate: float,
                      weight_decay: float,
                      encoder_weight_decay: float,
                      bias_weight_decay: bool,
                      **kwargs) -> Optimizer:
        """
        Initialize the optimizer

        Args:
        ----------
            optimizer_name (str):
                Name of the optimize. In-built torch optims and torch_optimizer lib 
                optimizers can be used.
            lookahead (bool):
                Flag whether the optimizer uses lookahead.
            model (nn.Module):
                pytorch model specification
            learning_rate (float):
                Decoder learning rate.
            weight_decay (float):
                decoder weight decay
            encoder_weight_decay (float):
                encoder weight decay
            bias_weight_decay (bool):
                Flag whether to apply weight decay for biases.
        """
        c = cls(
            model=model,
            learning_rate=learning_rate,
            encoder_learning_rate=encoder_learning_rate,
            weight_decay=weight_decay,
            encoder_weight_decay=encoder_weight_decay,
            bias_weight_decay=bias_weight_decay
        )

        optimz = list(optims.OPTIM_LOOKUP.keys())
        assert optimizer_name in optimz, f"optimizer: {optimizer_name} not one of {optimz}"

        kwargs = kwargs.copy()
        kwargs["lr"] = learning_rate
        kwargs["weight_decay"] = weight_decay
        kwargs["params"] = c.adjust_optim_params()

        key = optims.OPTIM_LOOKUP[optimizer_name]
        optimizer = optims.__dict__[key](**kwargs)

        if lookahead:
            optimizer = optims.Lookahead(optimizer, k=5, alpha=0.5)

        return optimizer