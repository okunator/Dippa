import torch
import torch.nn as nn
from typing import List

from ...modules import DenseConvBlock, DenseConvBlockPreact


class MultiBlockDense(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            n_blocks: int=2,
            preactivate: bool=False
        ) -> None:
        """
        Stacks dense conv blocks in a ModuleDict. These are used in the
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
                Normalization method. One of "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of "relu", "swish", "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            n_blocks (int, default=2):
                Number of BasicConvBlocks used in this block
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
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
        for _, conv_block in self.items():
            features = conv_block(features)
        return features