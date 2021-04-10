import torch
import torch.nn as nn
from typing import Tuple, List

from .block import MultiBlockBasic
from ..base_decoder_block import BaseDecoderBlock


class BasicDecoderBlock(BaseDecoderBlock):
    def __init__(self,
                 decoder_channels: List[int],
                 skip_channels: List[int],
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 up_sampling: str="fixed_unpool",
                 long_skip: str="unet",
                 long_skip_merge_policy: str="summation",
                 n_layers: int=1,
                 n_blocks: int=2,
                 preactivate: bool=False,
                 skip_index: int=None,
                 out_dims: List[int]=None) -> None:

        """
        Basic decoder block. 
        
        Operations:
        1. Upsample
        2. Long skip from encoder to decoder if specified.
        3. Convolve

        Args:
        -----------
            decoder_channels (List[int]):
                List of the Number of consecutive input channels in the decoder branch 
            skip_channels (List[int]):
                List of the number of channels in the encoder skip tensors.
                Ignored if long_skip == None.
                of the decoder blocks 
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str, default="relu"):
                Activation method. One of ("relu", "swish". "mish")
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            up_sampling (str, default="fixed_unpool"):
                up sampling method to be used.
                One of ("interp", "max_unpool", "transconv", "fixed_unpool")
            long_skip (str, default="unet"):
                long skip connection style to be used.
                One of ("unet", "unet++", "unet3+", None)
            long_skip_merge_policy (str, default: "cat):
                whether long skip is summed or concatenated
                One of ("summation", "concatenate")
            n_layers (int, default=1):
                The number of residual multiconv blocks inside one residual 
                decoder block
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one residual
                multiconv block        
            preactivate (bool, default=False)
                If True, normalization and activation are applied before convolution
            skip_index (int, default=None):
                the index of the skip_channels list. Used if long_skip="unet"
            out_dims (List[int], default=None):
                List of the heights/widths of each encoder/decoder feature map
                e.g. [256, 128, 64, 32, 16]. Assumption is that feature maps are
                square. This is used for skip blocks (unet3+, unet++)
        """
        super(BasicDecoderBlock, self).__init__(
            decoder_channels=decoder_channels,
            skip_channels=skip_channels,
            up_sampling=up_sampling,
            long_skip=long_skip,
            long_skip_merge_policy=long_skip_merge_policy,
            skip_index=skip_index,
            out_dims=out_dims,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=preactivate,
            n_blocks=1,
        )

        # multi conv blocks
        self.conv_modules = nn.ModuleDict()
        for i in range(n_layers):

            num_in_features = decoder_channels[skip_index + 1]
            if i == 0 and long_skip == "unet":
                num_in_features = self.in_channels
            elif i == 0 and long_skip == "unet++":
                # FIX THE FIRST CAHNNEL NUM
                print(skip_channels[skip_index])
                num_in_features = decoder_channels[skip_index]
            print(num_in_features)

            layer = MultiBlockBasic(
                in_channels=num_in_features, 
                out_channels=decoder_channels[skip_index + 1],
                n_blocks=n_blocks,
                batch_norm=batch_norm, 
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate
            )
            self.conv_modules[f"multiconv_block{i + 1}"] = layer
        
    def forward(self, x: torch.Tensor, skips: Tuple[torch.Tensor], idx: int=None) -> torch.Tensor:
        """
        Args:
        ----------
            x (torch.Tensor):
                Input tensor. Shape (B, C, H, W).
            skips (Tuple[torch.Tensor]):
                Tuple of tensors generated from consecutive encoder blocks.
                Shapes (B, C, H, W).
            idx (int, default=None):
                runnning index used to get the right skip tensor(s) from the skips
                Tuple for the skip connection.
        """
        # upsample and long skip
        x = self.upsample(x)
        if self.skip is not None:
            x = self.skip(x, skips, idx=idx)
        
        # basic conv blocks
        for name, module in self.conv_modules.items():
            x = module(x)
        return x
