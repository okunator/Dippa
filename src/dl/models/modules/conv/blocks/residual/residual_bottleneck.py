import torch
import torch.nn as nn

from .._base._base_bottleneck import BaseBottleneckConv
from ...ops.utils import conv_func
from ....normalization.utils import norm_func


class BottleneckResidual(BaseBottleneckConv):
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
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Bottleneck Residual conv block that can be used to build deep
        residual bottleneck layers.

        Residual connection applied before the final activation.

        Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        ---------------------------------------------------------
        |                                                       |
        Input -> C1 -> N1 -> ACT -> C2 -> N2 -> ACT -> C3 -> N3 + -> ACT

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
            pre_attend=True,
            preactivate=False,
            **kwargs
        )

        # Set downsampling to enable residual summation
        self.downsample = None
        if in_channels != out_channels*self.expansion:
            self.downsample = nn.Sequential(
                conv_func(
                    self.conv_choice, in_channels=in_channels,
                    bias=False, out_channels=out_channels*self.expansion,
                    kernel_size=1, padding=0
                ),
                norm_func(
                    normalization, num_features=out_channels*self.expansion
                )
            ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
 
        # shortcut
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)

        # pre-attention
        out = self.attend(x)

        # residual
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out += identity
        out = self.act3(out)

        return out


class BottleneckResidualPreact(BaseBottleneckConv):
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
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Preactivated bottleneck residual conv block that can be used to 
        build deep preactivated residual bottleneck layers.

        Residual connection applied before the final activation.

        Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027

        ---------------------------------------------------------
        |                                                       |
        Input -> N1 -> ACT -> C1 -> N2 -> ACT -> C1 -> N3 -> C1 + -> ACT

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            kernel_size (int, default=3):
                The size of the convolution kernel.
            expand_ratio (float, default=1.0):
                The ratio of channel expansion in the bottleneck
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
            pre_attend=True,
            preactivate=True,
            **kwargs
        )

        # Set downsampling to enable residual summation
        self.downsample = None
        if in_channels != out_channels*self.expansion:
            self.downsample = nn.Sequential(
                conv_func(
                    self.conv_choice, in_channels=in_channels, 
                    bias=False, out_channels=out_channels*self.expansion,
                    kernel_size=1, padding=0
                ),
                norm_func(
                    normalization, num_features=out_channels*self.expansion
                )
            )   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # shortcut
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # pre-attention
        out = self.attend(x)

        # residual
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.conv3(out)

        out += identity
        out = self.act3(out)

        return out
