import torch.nn as nn

from ._utils import make_divisible
from ...ops.utils import conv_func
from ....activations.utils import act_func
from ....normalization.utils import norm_func
from ....attention.utils import att_func


class BaseMBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float=1.0,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            preactivate: bool=False,
            kernel_size: int=3,
            attention: str=None,
        ) -> None:
        """
        Base MBconv block that is used in all decoder blocks.
        Inits the primitive conv block modules from the given arguments.

        MobileNetV2: Inverted Residuals and Linear Bottlenecks
            - https://arxiv.org/abs/1801.04381

        Args:
        -----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            expand_ratio (float, default=4.0):
                The ratio of channel expansion in the bottleneck
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
            attention (str, default=None):
                Attention method. One of: "se", None
        """
        super().__init__()
        self.conv_choice = "wsconv" if weight_standardize else "conv"
        self.out_channels = out_channels

        mid_channels = make_divisible(in_channels*expand_ratio)

        self.ch_pool = conv_func(
            name=self.conv_choice, in_channels=in_channels, padding=0,
            out_channels=mid_channels, kernel_size=1, bias=False
        )

        norm_channels = in_channels if preactivate else mid_channels
        self.norm1 = norm_func(normalization, num_features=norm_channels)
        self.act1 = act_func(activation)

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.depth_conv = conv_func(
            name=self.conv_choice, in_channels=mid_channels,
            out_channels=mid_channels, kernel_size=kernel_size, 
            groups=mid_channels, padding=padding, bias=False
        )
        self.norm2 = norm_func(normalization, num_features=mid_channels)
        self.act2 = act_func(activation)
        
        # set attention
        self.attend = att_func(
            attention, in_channels=mid_channels, squeeze_ratio=.04
        )

        self.proj_conv = conv_func(
            name=self.conv_choice, in_channels=mid_channels, bias=False,
            out_channels=self.out_channels, kernel_size=1, padding=0
        )

        norm_channels = mid_channels if preactivate else self.out_channels
        self.norm3 = norm_func(
            normalization, num_features=norm_channels
        )

