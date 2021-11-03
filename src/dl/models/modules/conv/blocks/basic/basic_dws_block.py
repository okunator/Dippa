import torch
import torch.nn as nn

from .basic_dws import DepthWiseSeparableBasic, DepthWiseSeparableBasicPreact


class DepthWiseSeparableBasicBlock(nn.ModuleDict):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        same_padding: bool=True,
        normalization: str="bn",
        activation: str="relu",
        weight_standardize: bool=False,
        n_blocks: int=2,
        preactivate: bool=False,
        attention: str=None
    ) -> None:
        """
        Stack dws basic blocks in a ModuleDict. These can be used
        in the full sized decoderblocks.

        Depthwise convs introduced:
            MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
                - https://arxiv.org/abs/1704.04861

        Preactivation introduced:
            Identity Mappings in Deep Residual Networks:
                - https://arxiv.org/abs/1603.05027

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
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
        super(DepthWiseSeparableBasicBlock, self).__init__()

        DWS = DepthWiseSeparableBasic
        if preactivate:
            DWS = DepthWiseSeparableBasicPreact

        for i in range(n_blocks):
            conv_block = DWS(
                in_channels=in_channels,
                out_channels=out_channels,
                same_padding=same_padding,
                normalization=normalization,
                activation=activation,
                weight_standardize=weight_standardize,
                attention=attention
            )
            self.add_module(f"depthwiseseparable{i + 1}", conv_block)
            in_channels = out_channels

        self.out_channels = in_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, conv_block in self.items():
            x = conv_block(x)
            
        return x