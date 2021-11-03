import torch.nn as nn

from ...ops.utils import conv_func
from ....activations.utils import act_func
from ....normalization.utils import norm_func
from ....attention.utils import att_func


class BaseConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            preactivate: bool=False,
            kernel_size=3,
            groups: int=1,
            attention: str=None,
            pre_attend: bool=False
        ) -> None:
        """
        Lightweight base conv block that can be used in decoder blocks.
        Inits the primitive conv block modules from the given arguments.

        I.e. Inits the Conv module, Norm module (optional) and Act
        module of any Conv block.

        Args:
        -----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
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
        super(BaseConv, self).__init__()
        conv_choice = "wsconv" if weight_standardize else "conv"

        # set norm channel number for preactivation or normal
        norm_channels = in_channels if preactivate else out_channels

        # set attention channels
        att_channels = in_channels if pre_attend else in_channels

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0
        
        self.conv = conv_func(
            name=conv_choice, in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, 
            groups=groups, padding=padding
        )

        self.norm = norm_func(normalization, num_features=norm_channels)      
        self.act = act_func(activation)
        self.attend = att_func(attention, in_channels=att_channels)
