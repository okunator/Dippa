import torch
import torch.nn as nn

from .residual import ResidualConvBlockPreact, ResidualConvBlock


class ResidualBlock(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
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
        Stack residual conv blocks in a ModuleDict. These are used in 
        the full sized decoderblocks.

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
        super().__init__()

        # use either preact or normal (original) resblock
        Residual = ResidualConvBlock
        if preactivate:
            Residual = ResidualConvBlockPreact

        blocks = list(range(n_blocks))
        for i in blocks:
            att_method = attention if i == blocks[-1] else None
            conv_block = Residual(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                same_padding=same_padding,
                normalization=normalization,
                activation=activation,
                weight_standardize=weight_standardize,
                attention=att_method if attention is not None else None
            )
            self.add_module(f"conv{i + 1}", conv_block)
            in_channels = conv_block.out_channels

        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, conv_block in self.items():
            x = conv_block(x)
            
        return x
