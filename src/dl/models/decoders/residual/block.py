import torch
import torch.nn as nn

import src.dl.models.layers.activations as act
import src.dl.models.layers.normalization as norm
from src.dl.models.decoders.base_block import BaseConvBlock


class ResidualConvBlock(BaseConvBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 use_residual: bool=True) -> None:
        """
        Residual conv block that can be used in decoders.
        Residual connection applied before the final activation.

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
            use_residual (bool, default=True):
                If True, the identity is summed to the linear unit 
                before the final activation
        """
        super(ResidualConvBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=False
        )
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv_choices[self.conv_choice](x)
        x = self.bn_choices[self.batch_norm](x)

        if self.use_residual:
            x += identity

        x = self.act_choices[self.activation](x)
        return x


class MultiBlockResidual(nn.ModuleDict):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 n_blocks: int=2) -> None:
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
        """
        super(MultiBlockResidual, self).__init__()
        self.conv1 = ResidualConvBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            same_padding=same_padding,
            batch_norm=batch_norm, 
            activation=activation, 
            weight_standardize=weight_standardize,
            use_residual=False
        )

        blocks = list(range(1, n_blocks))
        for i in blocks:
            # apply residual connection at the final conv block
            use_residual = i == blocks[-1]
            conv_block = ResidualConvBlock(
                in_channels=out_channels, 
                out_channels=out_channels, 
                same_padding=same_padding, 
                batch_norm=batch_norm, 
                activation=activation, 
                weight_standardize=weight_standardize,
                use_residual=use_residual
            )
            self.add_module('conv%d' % (i + 1), conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, conv_block in self.items():
            x = conv_block(x)
        return x