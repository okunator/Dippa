import torch
import torch.nn as nn

import src.dl.activations as act
from src.dl.modules import WSConv2d
from src.dl.normalization import EstBN, BCNorm


class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool = True,
                 batch_norm: str = "bn",
                 activation: str = "relu",
                 weight_standardize: bool = False) -> None:
        """
        Basic conv block that can be used in decoders

        Args:
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool):
                if True, performs same-covolution
            batch_norm (str): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str):
                Activation method. One of (relu, swish. mish)
            weight_standardize (bool):
                If True, perform weight standardization
        """
        super(BasicConvBlock, self).__init__()
        assert batch_norm in ("bn", "bcn", None)
        assert activation in ("relu", "mish", "swish")
        
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv_choice = "wsconv" if weight_standardize else "conv"
        
        self.conv_choices = nn.ModuleDict({
            "wsconv": WSConv2d(in_channels, out_channels, kernel_size=3, padding=int(same_padding)),
            "conv": nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(same_padding))
        })

        self.bn_choices = nn.ModuleDict({
            "bn": nn.BatchNorm2d(num_features=out_channels),
            "bcn": BCNorm(num_features=out_channels, num_groups=32),
            "nope": nn.Identity()
        })

        self.act_choices = nn.ModuleDict({
            "relu": nn.ReLU(inplace=True),
            "mish": act.Mish(),
            "swish": act.Swish()
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_choices[self.conv_choice](x)
        x = self.bn_choices[self.batch_norm](x)
        x = self.act_choices[self.activation](x)
        return x
