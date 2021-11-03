import torch

from .._base._base_depthwise import BaseDepthWiseSeparableConv


class DepthWiseSeparableBasic(BaseDepthWiseSeparableConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Depthwise separable basic conv block that can be used to 
        build deep dws layers.

        Depthwise convs introduced:
            MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
                - https://arxiv.org/abs/1704.04861


        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
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
        super(DepthWiseSeparableBasic, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            preactivate=False,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise conv
        out = self.depth_conv(x)
        out = self.norm1(out)
        out = self.act(out)

        # attention
        out = self.attend(out)

        # pointwise channel pool
        out = self.ch_pool(out)
        out = self.norm2(out)
        out = self.act(out)

        return out


class DepthWiseSeparableBasicPreact(BaseDepthWiseSeparableConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Depthwise separable basic preactivated conv block that can be
        used to build deep dws layers with preactivation.

        Depthwise convs introduced:
            MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
                - https://arxiv.org/abs/1704.04861


        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
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
        super(DepthWiseSeparableBasicPreact, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            preactivate=True,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise conv
        out = self.norm1(x)
        out = self.act(out)
        out = self.depth_conv(out)

        # attention
        out = self.attend(out)

        # pointwise channel pool
        out = self.norm2(out)
        out = self.ch_pool(out)
        out = self.act(out)

        return out
        