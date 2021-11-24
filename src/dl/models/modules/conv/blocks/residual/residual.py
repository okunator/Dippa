import torch
import torch.nn as nn

from .._base._base_conv import BaseConv
from ...ops.utils import conv_func
from ....normalization.utils import norm_func


class ResidualConvBlock(BaseConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        A simplified residual conv block that can be used to build deep
        residual layers,

        Residual connection applied before the final activation.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385
        
         ----------------------
        |                      |
        Input -> CONV -> BN -> + -> RELU -> OUTPUT

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            kernel_size (int, default=3):
                The size of the convolution kernel.
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
            attention (str, default=None):
                Attention method. One of: "se", None
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            pre_attend=True,
            preactivate=False,
            **kwargs
        )

        # Set downsampling to enable residual summation
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv_func(
                    self.conv_choice, in_channels=in_channels,
                    bias=False, out_channels=out_channels,
                    kernel_size=1, padding=0
                ),
                norm_func(
                    normalization, num_features=out_channels
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # pre-attention
        x = self.attend(x)

        # residual
        x = self.conv(x)
        x = self.norm(x)

        x += identity
        x = self.act(x)

        return x


class ResidualConvBlockPreact(BaseConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str=None,
        ) -> None:
        """
        A simplified Preactivated Residual conv block that can be used
        to build deep residual layers

        Residual connection applied after the final conv.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        (Preact-ResNet): Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027
        
         ------------------------------
        |                              |
        Input -> BN -> RELU -> CONV -> + -> OUTPUT

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            kernel_size (int, default=3):
                The size of the convolution kernel.
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
            attention (str, default=None):
                Attention method. One of: "se", None
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            pre_attend=True,
            preactivate=True
        )

        # Set downsampling to enable residual summation
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv_func(
                    self.conv_choice, in_channels=in_channels,
                    bias=False, out_channels=out_channels,
                    kernel_size=1, padding=0
                ),
                norm_func(
                    normalization, num_features=out_channels
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # pre-attention
        x = self.attend(x)

        # preact residual
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)

        x += identity
        x = self.act(x)

        return x