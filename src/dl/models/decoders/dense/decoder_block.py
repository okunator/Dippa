import torch
import torch.nn as nn
from typing import Tuple, List

from .block import MultiBlockDense, DenseConvBlock, DenseConvBlockPreact
from ..base_decoder_block import BaseDecoderBlock


class DenseDecoderBlock(BaseDecoderBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
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
        Dense decoder block. 
        
        Operations:
        1. Upsample
        2. Long skip from encoder to decoder if specified.
        3. Convolve + dense short skip connectin

        Args:
        -----------
            in_channels (int):
                Number of input channels
            out_channels (List[int]):
                List of the number of output channels in the decoder output tensors 
            skip_channels (List[int]):
                List of the number of channels in the encoder skip tensors.
                Ignored if long_skip == None.
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
            long_skip_merge_policy (str, default="cat):
                whether long skip is summed or concatenated
                One of ("summation", "concatenate")
            n_layers (int, default=1):
                The number of dense multiconv blocks inside one dense decoder block
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one dense multiconv block
            preactivate (bool, default=False)
                If True, normalization and activation are applied before convolution
            skip_index (int, default=None):
                the index of the skip_channels list. Used if long_skip="unet"
            out_dims (List[int]):
                List of the heights/widths of each encoder/decoder feature map
                e.g. [256, 128, 64, 32, 16]. Assumption is that feature maps are
                square. This is used for skip blocks (unet3+, unet++)
        """
        super(DenseDecoderBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
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

       # layered multi conv blocks
        self.conv_modules = nn.ModuleDict()
        num_in_features = self.in_channels if long_skip == "unet" else out_channels[skip_index]
        for i in range(n_layers):
            num_out_features = out_channels[skip_index] # num_in_features // 2
            layer = MultiBlockDense(
                in_channels=num_in_features, 
                out_channels=num_out_features,
                n_blocks=n_blocks,
                batch_norm=batch_norm, 
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate
            )
            self.conv_modules[f"multiconv_block{i + 1}"] = layer
            num_in_features += num_out_features

        # Transition block to transition to a new decoder block
        DenseBlock = DenseConvBlockPreact if preactivate else DenseConvBlock
        self.transition = DenseBlock(
            in_channels=num_in_features, 
            out_channels=out_channels[skip_index], 
            same_padding=same_padding,
            batch_norm=batch_norm, 
            activation=activation, 
            weight_standardize=weight_standardize,
        )
        
    def forward(self, x: torch.Tensor, skips: Tuple[torch.Tensor], idx: int=None) -> torch.Tensor:
        """
        Args:
        -----------
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

        # dense blocks and append to feature list
        features = [x]
        for name, module in self.conv_modules.items():
            new_features = module(features)
            features.append(new_features)

        cat_features = torch.cat(features, dim=1)
        out = self.transition(cat_features)

        return out
