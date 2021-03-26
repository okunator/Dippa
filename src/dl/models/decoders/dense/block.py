import torch
import torch.nn as nn
from typing import Tuple, List

import src.dl.models.layers.activations as act
import src.dl.models.layers.normalization as norm
from src.dl.models.decoders.base_conv_block import BaseConvBlock


class DenseConvBlockPreact(BaseConvBlock):
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
        super(DenseConvBlockPreact, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        x = self.bn_choices[self.batch_norm](x)
        x = self.act_choices[self.activation](x)
        x = self.conv_choices[self.conv_choice](x)
        return x


class DenseConvBlock(BaseConvBlock):
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
        super(DenseConvBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        x = self.conv_choices[self.conv_choice](x)
        x = self.bn_choices[self.batch_norm](x)
        x = self.act_choices[self.activation](x)
        return x


class MultiBlockDense(nn.ModuleDict):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 n_blocks: int=2,
                 preactivate: bool=False) -> None:
        """
        Stack residual conv blocks in a ModuleDict. These are used in the
        full sized decoderblocks. The number of basic conv blocks can be 
        adjusted. Default is 2. The residual connection is applied at the
        final conv block, before the last activation.

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
        super(MultiBlockDense, self).__init__()

        DenseBlock = DenseConvBlockPreact if preactivate else DenseConvBlock

        # apply cat at the beginning of the block
        self.conv1 = DenseBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            same_padding=same_padding,
            batch_norm=batch_norm, 
            activation=activation, 
            weight_standardize=weight_standardize,
        )

        for i in range(1, n_blocks):
            conv_block = DenseBlock(
                in_channels=out_channels, 
                out_channels=out_channels, 
                same_padding=same_padding, 
                batch_norm=batch_norm, 
                activation=activation, 
                weight_standardize=weight_standardize,
            )
            self.add_module(f"conv{i + 1}", conv_block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        for name, conv_block in self.items():
            features = conv_block(features)
        return features