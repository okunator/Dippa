import torch.nn as nn

from ...ops.utils import conv_func
from ....activations.utils import act_func
from ....normalization.utils import norm_func
from ....attention.utils import att_func


class BaseDepthWiseSeparableConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            preactivate: bool=False,
            kernel_size: int=3,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Base depthwise separable conv block that can be used in decoder 
        blocks. Inits the primitive conv block modules from the given
        arguments.

        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications:
            - https://arxiv.org/abs/1704.04861

        Args:
        -----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            base_width (int, default=64):
                The minimum width for the conv out channels in this 
                block
            same_padding (bool):
                if True, performs same-covolution
            normalization (str): 
                Normalization method to be used.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            activation (str):
                Activation method. One of: "mish", "swish", "relu",
                "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh",
                "sigmoid", "silu", "prelu", "leaky-relu", "elu",
                "hardshrink", "tanhshrink", "hardsigmoid"
            weight_standardize (bool):
                If True, perform weight standardization
            preactivate (bool, default=False):
                If True, inits batch norm such that it will be
                applied before the convolution.
            kernel_size (int, default=3):
                The size of the convolution kernel.
            groups (int, default=1):
                Number of groups the kernels are divided into. If 
                `groups == 1` normal convolution is applied. If
                `groups = in_channels` depthwise convolution is applied
            attention (str, default=None):
                Attention method. One of: "se", None
        """
        super().__init__()
        self.conv_choice = "wsconv" if weight_standardize else "conv"
        self.out_channels = out_channels

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0

        self.depth_conv = conv_func(
            name=self.conv_choice, in_channels=in_channels,
            out_channels=in_channels, kernel_size=kernel_size, 
            groups=in_channels, padding=padding
        )

        self.norm1 = norm_func(normalization, num_features=in_channels)
        self.act1 = act_func(activation)
        self.attend = att_func(attention, in_channels=in_channels)

        self.ch_pool = conv_func(
            name=self.conv_choice, in_channels=in_channels, padding=0,
            out_channels=self.out_channels, kernel_size=1, bias=False,
        )

        norm_channels = in_channels if preactivate else self.out_channels
        self.norm2 = norm_func(normalization, num_features=norm_channels)
        self.act2 = act_func(activation)