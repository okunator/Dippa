import torch.nn as nn

from ...ops.utils import conv_func
from ....activations.utils import act_func
from ....normalization.utils import norm_func
from ....attention.utils import att_func


class BaseBottleneckConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float=4.0,
            base_width: int=64,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            preactivate: bool=False,
            kernel_size: int=3,
            groups: int=1,
            attention: str=None,
            pre_attend: bool=False
        ) -> None:
        """
        Base bottleneck conv block that can be used in decoder blocks.
        Inits the primitive conv block modules from the given arguments.

        (Res-Net): Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        (Preact-ResNet): Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027

        Args:
        -----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            expand_ratio (float, default=4.0):
                The ratio of channel expansion in the bottleneck
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
            pre_attend (bool, default=False):
                If True, Attention is applied at the beginning of
                forwards.
        """
        super(BaseBottleneckConv, self).__init__()
        self.conv_choice = "wsconv" if weight_standardize else "conv"
        self.expansion = int(expand_ratio)

        width = int(out_channels*(base_width / 64.0))*groups
        self.out_channels = out_channels*self.expansion

        # set attention channels
        att_channels = in_channels if pre_attend else self.out_channels
        self.attend = att_func(attention, in_channels=att_channels)

        self.conv1 = conv_func(
            name=self.conv_choice, in_channels=in_channels,
            out_channels=width, kernel_size=1, bias=False, padding=0
        )

        norm_channels = in_channels if preactivate else width
        self.norm1 = norm_func(normalization, num_features=norm_channels)
        self.act1 = act_func(activation)     

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.conv2 = conv_func(
            name=self.conv_choice, in_channels=width,
            out_channels=width, kernel_size=kernel_size, 
            groups=groups, padding=padding, bias=False
        )

        self.norm2 = norm_func(normalization, num_features=width)
        self.act2 = act_func(activation)     

        self.conv3 = conv_func(
            name=self.conv_choice, in_channels=width, bias=False,
            out_channels=self.out_channels, kernel_size=1, padding=0
        )

        norm_channels = width if preactivate else self.out_channels
        self.norm3 = norm_func(
            normalization, num_features=norm_channels
        )

        self.act3 = act_func(activation)     
