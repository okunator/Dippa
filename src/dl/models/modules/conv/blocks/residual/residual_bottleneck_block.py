import torch
import torch.nn as nn

from .residual_bottleneck import BottleneckResidual, BottleneckResidualPreact


class BottleneckResidualBlock(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float=4.0,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            n_blocks: int=2,
            preactivate: bool=False,
            attention: str=None
        ) -> None:
        """
        Stack residual bottleneck blocks in a ModuleDict. These can be 
        used in the full sized decoderblocks.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        (Preact-ResNet): Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            expand_ratio (float, default=1.0):
                The ratio of channel expansion in the bottleneck
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
            attention (str, default=None):
                Attention method. One of: "se", None
        """
        super(BottleneckResidualBlock, self).__init__()
        
        Bottleneck = BottleneckResidual
        if preactivate:
            Bottleneck = BottleneckResidualPreact

        for i in range(n_blocks):
            conv_block = Bottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                expand_ratio=expand_ratio,
                same_padding=same_padding,
                normalization=normalization,
                activation=activation,
                weight_standardize=weight_standardize,
                attention=attention
            )
            self.add_module(f"bottleneck{i + 1}", conv_block)
            in_channels = out_channels*conv_block.expansion

        self.out_channels = in_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, conv_block in self.items():
            x = conv_block(x)
            
        return x
