import torch
import torch.nn as nn
from typing import Tuple, List

from .multi_block import MultiBlockBasic
from ..base_decoder_block import BaseDecoderBlock


class BasicDecoderBlock(BaseDecoderBlock):
    def __init__(
            self,
            in_channels: int,
            out_channel_list: List[int],
            skip_channel_list: List[int],
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            up_sampling: str="fixed_unpool",
            short_skip: str=None,
            long_skip: str="unet",
            long_skip_merge_policy: str="summation",
            n_layers: int=1,
            n_blocks: int=2,
            preactivate: bool=False,
            skip_index: int=None,
            out_dims: List[int]=None
        ) -> None:

        """
        Basic decoder block. 
        
        Operations:
        1. Upsample
        2. Long skip from encoder to decoder if specified.
        3. Convolve

        Args:
        -----------
            in_channels (int):
                The number of channels coming in from the previous head
                or decoder branch
            out_channel_list (List[int]):
                List of the number of decoder branch output channels.
                First index contains the number of head channels.
            skip_channel_list (List[int]):
                List of the number of channels in the encoder skip 
                tensors. Ignored if long_skip == None.
            same_padding (bool, default=True):
                if True, performs same-covolution
            normalization (str): 
                Normalization method to be used.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            activation (str):
                Activation method. One of: "mish", "swish", "relu",
                "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh",
                "sigmoid", "silu", "prelu", "leaky-relu", "elu",
                "hardshrink", "tanhshrink", "hardsigmoid"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            up_sampling (str, default="fixed_unpool"):
                Up sampling method to be used. One of: "interp", 
                "max_unpool", "transconv",  "fixed_unpool"
            short_skip (str, default=None):
                Use short skip connections inside the skip blocks.
                One of ("resdidual", "dense", None)
            long_skip (str, default="unet"):
                long skip connection style to be used. One of: "unet", 
                "unet++", "unet3+", None
            long_skip_merge_policy (str, default: "cat):
                whether long skip is summed or concatenated One of: 
                "summation", "concatenate"
            n_layers (int, default=1):
                The number of multiconv blocks inside one decoder block
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one
                multiconv block        
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            skip_index (int, default=None):
                the index of the skip_channel_list list. Used if 
                `long_skip`="unet"
            out_dims (List[int], default=None):
                List of the heights/widths of each encoder/decoder 
                feature map e.g. [256, 128, 64, 32, 16]. Assumption is
                that feature maps are square. This is used for unet3+ 
                skip blocks
        """
        super(BasicDecoderBlock, self).__init__(
            in_channels=in_channels,
            out_channel_list=out_channel_list,
            skip_channel_list=skip_channel_list,
            up_sampling=up_sampling,
            short_skip=short_skip,
            long_skip=long_skip,
            long_skip_merge_policy=long_skip_merge_policy,
            skip_index=skip_index,
            out_dims=out_dims,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=preactivate,
            n_blocks=1,
        )
        # multi conv blocks
        self.conv_modules = nn.ModuleDict()
        for i in range(n_layers):

            # Set the number of input channels. Small kludge but what can u do..
            num_in_features = self.out_channels
            if i == 0:
                num_in_features = self.conv_in_channels

            layer = MultiBlockBasic(
                in_channels=num_in_features, 
                out_channels=self.out_channels,
                n_blocks=n_blocks,
                normalization=normalization,
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate
            )
            self.conv_modules[f"multiconv_block{i + 1}"] = layer
        
    def forward(
            self,
            x: torch.Tensor,
            ix: int,
            skips: Tuple[torch.Tensor],
            extra_skips: List[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Args:
        ----------
            x (torch.Tensor):
                Input tensor. Shape (B, C, H, W).
            ix (int, default=None):
                runnning index used to get the right skip tensor(s) from
                the skips tuple for the skip connection.
            skips (Tuple[torch.Tensor]):
                Tuple of tensors generated from consecutive encoder 
                blocks. Shapes (B, C, H, W).
            extra_skips (List[torch.Tensor], default=None):
                extra skip connections, Used in unet3+ and unet++
        """
        # upsample
        x = self.upsample(x)

        # long skip
        if self.skip is not None:
            x, extra = self.skip(
                x, ix=ix, skips=skips, extra_skips=extra_skips
            )
            extra_skips = extra

        # final basic conv blocks
        for _, module in self.conv_modules.items():
            x = module(x)

        return x, extra_skips
