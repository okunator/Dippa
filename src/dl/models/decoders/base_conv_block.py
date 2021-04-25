import torch
import torch.nn as nn

from ..modules import (
    Mish, Swish, BCNorm, WSConv2d, GroupNorm
)


class BaseConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 same_padding: bool=True,
                 batch_norm: str="bn",
                 activation: str="relu",
                 weight_standardize: bool=False,
                 preactivate: bool=False) -> None:
        """
        Base conv block that is used in all decoder blocks
        This uses moduledicts which let you choose the different methods
        to use.

        Args:
        -----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool):
                if True, performs same-covolution
            batch_norm (str): 
                Perform normalization. Methods:
                Batch norm, batch channel norm, group norm, etc.
                One of ("bn", "bcn", None)
            activation (str):
                Activation method. One of (relu, swish. mish)
            weight_standardize (bool):
                If True, perform weight standardization
            preactivate (bool, default=False):
                If True, inits batch norm such that it will be
                applied before the convolution.
        """
        super(BaseConvBlock, self).__init__()
        assert batch_norm in ("bn", "bcn", "gn", None)
        assert activation in ("relu", "mish", "swish", "leaky-relu")
        
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv_choice = "wsconv" if weight_standardize else "conv"

        # set norm channel number for preactivation or normal 
        bn_channels = in_channels if preactivate else out_channels
        
        # set convolution module
        if self.conv_choice == "wsconv":
            self.conv = WSConv2d(in_channels, out_channels, kernel_size=3, padding=int(same_padding))
        elif self.conv_choice == "conv":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(same_padding))

        # set normalization module
        if self.batch_norm == "bn":
            self.bn = nn.BatchNorm2d(num_features=bn_channels)
        elif self.batch_norm == "bcn":
            self.bn = BCNorm(num_features=bn_channels, num_groups=32)
        elif self.batch_norm == "gn":
            self.bn = GroupNorm(num_features=bn_channels, num_groups=32)
        else:
            self.bn = nn.Identity()
            
        # set activation module
        if self.activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.activation == "mish":
            self.act = Mish()
        elif self.activation == "swish":
            self.act = Swish()
        elif self.activation == "leaky-relu":
            self.act = nn.LeakyReLU(inplace=True) # slope = 0.1
