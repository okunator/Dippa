import torch
import torch.nn as nn
from typing import List

from .dense_fusedmbconv import FusedInvertedDense, FusedInvertedDensePreact
from ...ops.utils import conv_func
from ....activations.utils import act_func
from ....normalization.utils import norm_func


class FusedMobileInvertedDenseBlock(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float=4.0,
            kernel_size: int=3,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            n_blocks: int=2,
            preactivate: bool=False,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Stacks dense mbconv blocks in a ModuleDict. These can be used
        in the full sized decoder stages.

        DenseNet: Densely Connected Convolutional Networks
            - https://arxiv.org/abs/1608.06993
        
        Efficientnet-edgetpu: Creating accelerator-optimized neural networks with automl.
            - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

        EfficientNetV2: Smaller Models and Faster Training
            - https://arxiv.org/abs/2104.00298

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            expand_ratio (float, default=4.0):
                The ratio of channel expansion in the bottleneck
            kernel_size (int, default=3):
                The size of the convolution kernel.
            expand_ratio (float, default=1.0):
                The ratio of channel expansion in the bottlenec
            same_padding (bool, default=True):
                If True, performs same-covolution 
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
            n_blocks (int, default=2):
                Number of BasicConvBlocks used in this block
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
            attention (str, default=None):
                Attention method. One of: "se", None
        """
        super().__init__()
        
        Dense = FusedInvertedDense
        if preactivate:
            Dense = FusedInvertedDensePreact

        blocks = list(range(n_blocks))
        for i in blocks:
            att_method = attention if i == blocks[-1] else None
            conv_block = Dense(
                in_channels=in_channels,
                out_channels=out_channels,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                same_padding=same_padding,
                normalization=normalization,
                activation=activation,
                weight_standardize=weight_standardize,
                attention=att_method if attention is not None else None
            )
            self.add_module(f"dense_fusedmbconv{i + 1}", conv_block)
            in_channels += conv_block.out_channels
            
        # self.out_channels = in_channels
        
        self.transition = nn.Sequential(
            conv_func(
                conv_block.conv_choice, in_channels=in_channels,
                bias=False, out_channels=out_channels,
                kernel_size=1, padding=0
            ),
            norm_func(
                normalization, num_features=out_channels
            ),
            act_func(activation) 
        )
            
        self.out_channels = out_channels

    def forward(self, init_features: List[torch.Tensor]) -> torch.Tensor:
        features = [init_features]
        for name, conv_block  in self.items():
            if name is not "transition":            
                new_features = conv_block(features)
                features.append(new_features)
            
        out = torch.cat(features, 1)
        out = self.transition(out)
        
        return out
