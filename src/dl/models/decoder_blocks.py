import torch
import torch.nn as nn

import src.dl.activations as act
from src.dl.modules import WSConv2d, EstBN, BCNorm


class BasicConvBlock(nn.ModuleDict):
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
        
        # Normal conv or ws-conv
        if weight_standardize:
            wsconv = WSConv2d(
                in_channels, out_channels,
                kernel_size=3, padding=int(same_padding)
            )
            self.add_module("wsconv", wsconv)
        else:
            conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=int(same_padding)
            )
            self.add_module("conv", conv)

        # batch norm or batch channel norm
        if batch_norm == "bn":
            bn = nn.BatchNorm2d(num_features=out_channels)
            self.add_module("bn", bn)
        elif batch_norm == "bcn":
            bcn = BCNorm(num_features=out_channels)
            self.add_module("bcn", bcn)
        else:
            self.add_module("identity", nn.Identity())

        # activation function
        if activation == "relu":
            relu = nn.ReLU(inplace=True)
            self.add_module("relu", relu)
        elif activation == "swish":
            self.add_module("swish", act.Swish())
        elif activation == "mish":
            self.add_module("mish", act.Mish())

    def forward(self, x: torch.Tensor):
        return self.block(x)