import torch
from typing import Union, List

from .._base._base_depthwise import BaseDepthWiseSeparableConv


class DepthWiseSeparableDense(BaseDepthWiseSeparableConv):
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
        Depthwise separable dense conv block that can be used to 
        build deep dense dws layers.

        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
            - https://arxiv.org/abs/1704.04861

        DenseNet: Densely Connected Convolutional Networks
            - https://arxiv.org/abs/1608.06993

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

    def forward(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]]
        ) -> torch.Tensor:
 
        # dense skip
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        # depthwise conv
        out = self.depth_conv(x)
        out = self.norm1(out)
        out = self.act1(out)

        # attention
        out = self.attend(out)

        # pointwise channel pool
        out = self.ch_pool(out)
        out = self.norm2(out)
        out = self.act2(out)

        return out


class DepthWiseSeparableDensePreact(BaseDepthWiseSeparableConv):
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
        preactivated depthwise separable dense conv block that can be
        used to build deep preactivated dense dws layers.

        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
            - https://arxiv.org/abs/1704.04861

        DenseNet: Densely Connected Convolutional Networks
            - https://arxiv.org/abs/1608.06993

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
            preactivate=True,
            **kwargs
        )

    def forward(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]]
        ) -> torch.Tensor:
 
        # dense skip
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        # depthwise conv
        out = self.norm1(x)
        out = self.act1(out)
        out = self.depth_conv(out)

        # attention
        out = self.attend(out)

        # pointwise channel pool
        out = self.norm2(out)
        out = self.act2(out)
        out = self.ch_pool(out)

        return out