import torch
import torch.nn as nn
from typing import Tuple

import src.dl.models.layers as layers
from .block import MultiBlockBasic


class BasicDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 same_padding: bool = True,
                 batch_norm: str = "bn",
                 activation: str = "relu",
                 weight_standardize: bool = False,
                 n_blocks: int = 2,
                 up_sampling: str = "fixed_unpool",
                 long_skip: str = "nope") -> None:

        """
        Basic decoder block. 
        
        Operations:
        1. Upsample
        2. Long skip from encoder to decoder if specified.
        3. Convolve

        Args:
            in_channels (int):
                Number of input channels
            skip_channels (int):
                Number of channels in the encoder skip tensor
            out_channels (int):
                Number of output channels
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
            n_blocks (int, default=2):
                Number of basic convolution blocks in this Decoder block
            up_sampling (str, default="fixed_unpool"):
                up sampling method to be used.
                One of ("interp", "segnet", "transconv", "fixed_unpool")
            long_skip (str, default="unet"):
                long skip connection style to be used.
                One of ("unet", "unet++", "unet3+", "nope")            
        """
        super(BasicDecoderBlock, self).__init__()
        assert up_sampling in ("interp", "segnet", "transconv", "fixed_unpool")
        assert long_skip in ("unet", "unet++", "unet3p", "nope")
        
        self.up_sampling = up_sampling
        self.long_skip = long_skip
    
        self.up_choices = nn.ModuleDict({
            "fixed_unpool": layers.FixedUnpool(scale_factor=2),
            "interp": None,
            "transconv":None,
            "segnet":None
        })
        
        self.skip_choices = nn.ModuleDict({
            "unet": layers.UnetSkipBlock(),
            "unet++": None,
            "unet3+": None,
        })

        self.conv = MultiBlockBasic(
            in_channels=in_channels+skip_channels, 
            out_channels=out_channels,
            n_blocks=n_blocks,
            batch_norm=batch_norm, 
            activation=activation,
            weight_standardize=weight_standardize
        )
        
    def forward(self, x: torch.Tensor, skips: Tuple[torch.Tensor]=None) -> torch.Tensor:
        x = self.up_choices[self.up_sampling](x)
        if skips is not None:
            x = self.skip_choices[self.long_skip](x, skips)
        x = self.conv(x)

        return x
