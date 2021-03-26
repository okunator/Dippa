import torch
import torch.nn as nn

import src.dl.models.layers.activations as act
import src.dl.models.layers.normalization as norm
from src.dl.models.decoders.base_conv_block import BaseConvBlock


class BasicConvBlockPreact(BaseConvBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False) -> None:
        """
        Basic conv block that can be used in decoders

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str, default="relu"):
                Activation method. One of (relu, swish. mish)
            weight_standardize (bool, default=False):
                If True, perform weight standardization
        """
        super(BasicConvBlockPreact, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_choices[self.batch_norm](x)
        x = self.act_choices[self.activation](x)
        x = self.conv_choices[self.conv_choice](x)
        return x


class BasicConvBlock(BaseConvBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False) -> None:
        """
        Basic conv block that can be used in decoders

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str, default="relu"):
                Activation method. One of (relu, swish. mish)
            weight_standardize (bool, default=False):
                If True, perform weight standardization
        """
        super(BasicConvBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_choices[self.conv_choice](x)
        x = self.bn_choices[self.batch_norm](x)
        x = self.act_choices[self.activation](x)
        return x


class MultiBlockBasic(nn.ModuleDict):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool = True,
                 batch_norm: str = "bn",
                 activation: str = "relu",
                 weight_standardize: bool = False,
                 n_blocks: int = 2,
                 preactivate: bool=False) -> None:
        """
        Stack basic conv blocks in a ModuleDict. These are used in the
        full sized decoderblocks. The number of basic conv blocks can be 
        adjusted. Default is 2.

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                If True, performs same-covolution
            batch_norm (str, default="bn"): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str, default="relu"):
                Activation method. One of (relu, swish. mish)
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            n_blocks (int, default=2):
                Number of BasicConvBlocks used in this block
            preactivate (bool, default=False)
                If True, normalization and activation are applied before convolution
        """
        super(MultiBlockBasic, self).__init__()

        # use either preact or normal conv block
        ConvBlock = BasicConvBlockPreact if preactivate else BasicConvBlock

        self.conv1 = ConvBlock(
            in_channels=in_channels, 
            out_channels=out_channels,
            same_padding=same_padding, 
            batch_norm=batch_norm, 
            activation=activation, 
            weight_standardize=weight_standardize
        )

        for i in range(1, n_blocks):
            conv_block = ConvBlock(
                in_channels=out_channels, 
                out_channels=out_channels, 
                same_padding=same_padding, 
                batch_norm=batch_norm, 
                activation=activation, 
                weight_standardize=weight_standardize
            )
            self.add_module('conv%d' % (i + 1), conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, conv_block in self.items():
            x = conv_block(x)
        return x
