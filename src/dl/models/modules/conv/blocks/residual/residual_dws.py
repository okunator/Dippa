import torch
import torch.nn as nn

from .._base._base_depthwise import BaseDepthWiseSeparableConv
from ...ops.utils import conv_func
from ....normalization.utils import norm_func


class DepthWiseSeparableResidual(BaseDepthWiseSeparableConv):
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
        Depthwise separable residual conv block that can be used to 
        build deep residual dws layers.

        Residual connection applied before the final activation.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
            - https://arxiv.org/abs/1704.04861


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
        # shortcut
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # depthwise conv
        out = self.depth_conv(x)
        out = self.norm1(out)
        out = self.act1(out)

        # attention
        out = self.attend(out)

        # pointwise channel pool
        out = self.ch_pool(out)
        out = self.norm2(out)

        out += identity
        out = self.act2(out)

        return out


class DepthWiseSeparableResidualPreact(BaseDepthWiseSeparableConv):
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
        Preactivated depthwise separable residual conv block that can be
        used to build deep residual dws layers with preactivation.

        Residual connection applied before the final activation.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        (Preact-ResNet): Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027

        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
            - https://arxiv.org/abs/1704.04861


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
            same_padding=same_padding,
            kernel_size=kernel_size,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            preactivate=True,
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
        # shortcut
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # depthwise conv
        out = self.norm1(x)
        out = self.act1(out)
        out = self.depth_conv(out)

        # attention
        out = self.attend(out)

        # pointwise channel pool
        out = self.norm2(out)
        out = self.ch_pool(out)

        out += identity
        out = self.act2(out)

        return out
        