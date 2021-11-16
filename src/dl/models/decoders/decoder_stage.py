import torch
import torch.nn as nn
from typing import Tuple, List, Union

from .long_skips.utils import long_skip_func
from ..modules.conv.utils import conv_block_func
from ..modules.upsampling.utils import up_func


class DecoderStage(nn.Module):
    def __init__(
            self,
            stage_ix: int,
            dec_channels: List[int],
            skip_channels: List[int],
            dec_out_dims: List[int],
            conv_block_type: str="basic",
            short_skip: str=None,
            long_skip: str="unet",
            long_skip_merge_policy: str="summation",
            n_layers: int=1,
            n_blocks: int=2,
            same_padding=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            upsampling: str="fixed-unpool",
            preactivate: bool=False,
            attention: str=None
        ) -> None:

        """
        Abstraction for a decoder stage. 
        
        Operations in each decoder stage:
        1. Upsample
        2. Long skip from encoder to decoder if not the last stage.
        3. Conv block

        Args:
        -----------
            stage_ix (int):
                The index numer of the current decoder stage
            dec_channels (List[int]):
                List of the number of output channels in the decoder 
                output stages. First index contains the number of head 
                channels
            skip_channels (List[int]):
                List of the number of channels in the encoder skip 
                tensors. Ignored if `long_skip` == None.
            dec_out_dims (List[int], default=None):
                List of the heights/widths of each encoder/decoder 
                feature map e.g. [256, 128, 64, 32, 16]. Assumption is 
                that feature maps are square.
            conv_block_type (str, default="basic"):
                The type of the multi convolution blocks to be used.
                One of "basic", "bottleneck", "mbconv", "fusedmbconv", 
                "dws"
            n_layers (int, default=1):
                The number of mutli conv blocks inside one decoder stage 
            n_blocks (int, default=2):
                Number of conv-blocks inside one multi conv block
            short_skip (str, default=None):
                Use short skip connections inside the skip blocks.
                One of: "resdidual", "dense", None
            long_skip (str, default="unet"):
                long skip connection style to be used. One of: "unet", 
                "unet++", "unet3+", None
            long_skip_merge_policy (str, default: "cat):
                whether long skip is summed or concatenated. One of: 
                "summation", "concatenate"
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
            upsampling (str, default="fixed-unpool"):
                upsampling method to be used. One of "linear", "bicubic"
                "transconv", "fixed-unpool", "bilinear", "trilinear"
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            attention (str, default=None):
                Attention method used in the conv blocks. 
                One of: "se", None
        """
        super().__init__()
        assert n_layers >= 1
        assert n_blocks >= 1
        
        stage_ix = stage_ix
        in_channels = dec_channels[stage_ix]
        out_channels = dec_channels[stage_ix + 1]
        short_skip = short_skip if short_skip is not None else "basic"

        # upsampling method
        self.upsample = up_func(upsampling)

        # long skip connection method
        self.skip = long_skip_func(
            name=long_skip,
            stage_ix=stage_ix,
            in_channels=in_channels,
            dec_channels=dec_channels,
            skip_channels=skip_channels,
            dec_out_dims=dec_out_dims,
            short_skip=short_skip,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=preactivate,
            n_conv_blocks=n_blocks,
            merge_policy=long_skip_merge_policy,
            same_padding=same_padding
        )

        # Set up n layers of conv blocks 
        self.conv_modules = nn.ModuleDict()
        for i in range(n_layers):
            n_in_feats = self.skip.out_channels if i == 0 else in_channels
            layer = conv_block_func(
                name=conv_block_type,
                skip_type=short_skip,
                in_channels=n_in_feats, 
                out_channels=out_channels,
                n_blocks=n_blocks,
                normalization=normalization, 
                activation=activation,
                weight_standardize=weight_standardize,
                preactivate=preactivate,
                attention=attention,
                same_padding=same_padding
            )
            self.conv_modules[f"multiconv_block{i + 1}"] = layer
            in_channels = layer.out_channels
            
        self.out_channels = in_channels
        
    def forward(
            self,
            x: torch.Tensor,
            ix: int,
            skips: Tuple[torch.Tensor],
            extra_skips: List[torch.Tensor]=None
        ) -> Tuple[torch.Tensor, Union[None, List[torch.Tensor]]]:
        """
        Args:
        ----------
            x (torch.Tensor):
                Input tensor. Shape (B, C, H, W).
            ix (int):
                runnning index used to get the right skip tensor(s) 
                from the skips tuple for the skip connection.
            skips (Tuple[torch.Tensor]):
                All of feature maps from consecutive encoder blocks. 
                except the very first feature map that is not used in
                long skips. Order is bottom up. 
            extra_skips (List[torch.Tensor], default=None):
                extra skip connections, Used in unet3+ and unet++

        Returns:
        ----------
            Tuple: output torch.Tensor and extra skip torch.Tensors in 
                   a list. If no extra skips are present returns None 
                   instead of a list.
        """
        # upsample
        x = self.upsample(x)

        # long skip
        if self.skip is not None:
            x, extra = self.skip(
                x, ix=ix, skips=skips, extra_skips=extra_skips
            )
            extra_skips = extra
        
        # conv blocks
        for _, module in self.conv_modules.items():
            x = module(x)

        return x, extra_skips