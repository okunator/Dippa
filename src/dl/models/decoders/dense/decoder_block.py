import torch
import torch.nn as nn
from typing import Tuple

import src.dl.models.layers as layers
from .block import MultiBlockDense, DenseConvBlock


class DenseDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 up_sampling: str="fixed_unpool",
                 long_skip: str="unet",
                 long_skip_merge_policy: str="summation",
                 n_layers: int=1,
                 n_blocks: int=2) -> None:

        """
        Dense decoder block. 
        
        Operations:
        1. Upsample
        2. Long skip from encoder to decoder if specified.
        3. Convolve + dense short skips from the layers in this block

        Args:
        -----------
            in_channels (int):
                Number of input channels
            skip_channels (int):
                Number of channels in the encoder skip tensor.
                Ignored if long_skip == "nope".
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
            up_sampling (str, default="fixed_unpool"):
                up sampling method to be used.
                One of ("interp", "max_unpool", "transconv", "fixed_unpool")
            long_skip (str, default="unet"):
                long skip connection style to be used.
                One of ("unet", "unet++", "unet3+", "nope")
            long_skip_merge_policy (str, default="cat):
                whether long skip is summed or concatenated
                One of ("summation", "concatenate")
            n_layers (int, default=1):
                The number of dense multiconv blocks inside one dense decoder block
            n_blocks (int, default=2):
                Number of basic (bn->relu->conv)-blocks inside one dense multiconv block
        """
        super(DenseDecoderBlock, self).__init__()
        assert up_sampling in ("interp", "max_unpool", "transconv", "fixed_unpool")
        assert long_skip in ("unet", "unet++", "unet3+", "nope")
        assert long_skip_merge_policy in ("concatenate", "summation")

        self.up_sampling = up_sampling
        self.long_skip = None if long_skip == "nope" else long_skip
        self.merge_pol = long_skip_merge_policy

        # set upsampling choices
        self.up_choices = nn.ModuleDict({
            "fixed_unpool": layers.FixedUnpool(scale_factor=2),
            "interp": None,
            "transconv":None,
            "max_unpool":None
        })
        
        # Set skip connection block choices
        self.skip_choices = None
        if self.long_skip is not None:

            # adjust input channel dim if "concatenate"
            if self.merge_pol == "concatenate":
                in_channels += skip_channels

            self.skip_choices = nn.ModuleDict({
                "unet": layers.UnetSkipBlock(
                    merge_policy=self.merge_pol, 
                    skip_channels=skip_channels, 
                    in_channels=in_channels
                ),
                "unet++": None,
                "unet3+": None,
            })

        # layered multi conv blocks
        self.conv_modules = nn.ModuleDict()
        num_in_features = in_channels
        for i in range(n_layers):
            num_out_features = out_channels # num_in_features // 2 
            layer = MultiBlockDense(
                in_channels=num_in_features, 
                out_channels=num_out_features,
                n_blocks=n_blocks,
                batch_norm=batch_norm, 
                activation=activation,
                weight_standardize=weight_standardize
            )

            self.conv_modules[f"multiconv_block{i + 1}"] = layer
            num_in_features += num_out_features

        # Transition block for transition to a new decoder block
        self.transition = DenseConvBlock(
            in_channels=num_in_features, 
            out_channels=out_channels, 
            same_padding=same_padding,
            batch_norm=batch_norm, 
            activation=activation, 
            weight_standardize=weight_standardize,
        )
        
    def forward(self, x: torch.Tensor, skips: Tuple[torch.Tensor], **kwargs) -> torch.Tensor:
        x = self.up_choices[self.up_sampling](x)
        if self.skip_choices is not None:
            x = self.skip_choices[self.long_skip](x, skips, **kwargs)

        features = [x]
        for name, module in self.conv_modules.items():
            new_features = module(features)
            features.append(new_features)

        cat_features = torch.cat(features, dim=1)
        out = self.transition(cat_features)

        return out
