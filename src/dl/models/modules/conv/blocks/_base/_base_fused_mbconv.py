import torch.nn as nn

from ._utils import make_divisible
from ...ops.utils import conv_func
from ....activations.utils import act_func
from ....normalization.utils import norm_func
from ....attention.utils import att_func


class BaseFusedMBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int=1,
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
        Base FusedMBconv block that is used in all decoder blocks.
        Inits the primitive conv block modules from the given arguments.

        Efficientnet-edgetpu: Creating accelerator-optimized neural networks with automl.
            - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

        EfficientNetV2: Smaller Models and Faster Training
            - https://arxiv.org/abs/2104.00298

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
        assert stride in (1, 2)
        self.conv_choice = "wsconv" if weight_standardize else "conv"
        self.out_channels = out_channels

        mid_channels = make_divisible(in_channels*expand_ratio)

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.conv1 = conv_func(
            name=self.conv_choice, in_channels=in_channels, padding=padding,
            out_channels=mid_channels, kernel_size=kernel_size, bias=False,
            stride=stride
        )

        norm_channels = in_channels if preactivate else mid_channels
        self.norm1 = norm_func(normalization, num_features=norm_channels)

        # set attention
        self.attend = att_func(
            attention, in_channels=mid_channels, squeeze_ratio=.04
        )

        self.proj_conv = conv_func(
            name=self.conv_choice, in_channels=mid_channels, bias=False,
            out_channels=self.out_channels, kernel_size=1, padding=0
        )

        norm_channels = mid_channels if preactivate else self.out_channels
        self.norm2 = norm_func(
            normalization, num_features=norm_channels
        )

        self.act = act_func(activation)
