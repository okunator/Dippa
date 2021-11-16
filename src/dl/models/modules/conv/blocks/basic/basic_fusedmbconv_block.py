import torch
import torch.nn as nn

from .basic_fusedmbconv import FusedInvertedBasic, FusedInvertedBasicPreact


class FusedMobileInvertedBasicBlock(nn.ModuleDict):
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
        n_blocks: int=4,
        preactivate: bool=False,
        attention: str=None,
        **kwargs
    ) -> None:
        """
        Stack fused inverted basic blocks in a ModuleDict. These can 
        be used in the full sized decoderblocks.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        (Preact-ResNet): Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027

        MobileNetV2: Inverted Residuals and Linear Bottlenecks
            - https://arxiv.org/abs/1801.04381

        NOTE: 
        Basic in the name here means that no dense or residual skip 
        connections are applied in the forward method

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
        super(FusedMobileInvertedBasicBlock, self).__init__()

        FusedMBConv = FusedInvertedBasic
        if preactivate:
            FusedMBConv = FusedInvertedBasicPreact

        for i in range(n_blocks):
            conv_block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1, # no need to downsample when i == 0. Blocks not used for cls tasks
                expand_ratio=expand_ratio,
                same_padding=same_padding,
                normalization=normalization,
                activation=activation,
                weight_standardize=weight_standardize,
                attention=attention
            )
            self.add_module(f"fused_inverted_basic{i + 1}", conv_block)
            in_channels = conv_block.out_channels

        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, conv_block in self.items():
            x = conv_block(x)
            
        return x