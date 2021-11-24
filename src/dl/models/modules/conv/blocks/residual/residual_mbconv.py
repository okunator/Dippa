import torch
import torch.nn as nn

from .._base._base_mbconv import BaseMBConv
from ...ops.utils import conv_func
from ....normalization.utils import norm_func


class InvertedResidual(BaseMBConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            expand_ratio: float=4.0,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str="se",
            **kwargs
        ) -> None:
        """
        Mobile inverted residual conv block that can be used to 
        build deep residual dws layers.

        Residual connection applied before the final activation.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        MobileNetV2: Inverted Residuals and Linear Bottlenecks:
            - https://arxiv.org/abs/1801.04381


        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            expand_ratio (float, default=1.0):
                The ratio of channel expansion in the bottleneck
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
            expand_ratio=expand_ratio,
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

        # pointwise channel pooling conv
        out = self.ch_pool(x)
        out = self.norm1(out)
        out = self.act1(out)

        # depthwise conv
        out = self.depth_conv(out)
        out = self.norm2(out)
        out = self.act2(out)

        # attention
        out = self.attend(out)

        # Pointwise linear projection
        out = self.proj_conv(out)
        out = self.norm3(out)

        out += identity

        return out


class InvertedResidualPreact(BaseMBConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            expand_ratio: float=4.0,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str="se",
            **kwargs
        ) -> None:
        """
        Preactivated mobile inverted residual conv block that can be
        used to build deep residual dws layers with preactivation.

        Residual connection applied before the final activation.

        MobileNetV2: Inverted Residuals and Linear Bottlenecks:
            - https://arxiv.org/abs/1801.04381

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            expand_ratio (float, default=1.0):
                The ratio of channel expansion in the bottleneck
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
            expand_ratio=expand_ratio,
            kernel_size=kernel_size,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            preactivate=True,
            **kwargs
        )

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

        # pointwise channel pooling conv
        out = self.norm1(x)
        out = self.act1(out)
        out = self.ch_pool(out)

        # depthwise conv
        out = self.norm2(out)
        out = self.act2(out)
        out = self.depth_conv(out)

        # attention
        out = self.attend(out)

        # Pointwise linear projection
        out = self.norm3(out)
        out = self.proj_conv(out)

        out += identity

        return out
        