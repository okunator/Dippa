import torch
import torch.nn as nn

import src.dl.models.layers.activations as act
import src.dl.models.layers.normalization as norm


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
        assert batch_norm in ("bn", "bcn", None)
        assert activation in ("relu", "mish", "swish")
        
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv_choice = "wsconv" if weight_standardize else "conv"

        bn_channels = in_channels if preactivate else out_channels
        
        self.conv_choices = nn.ModuleDict({
            "wsconv": norm.WSConv2d(in_channels, out_channels, kernel_size=3, padding=int(same_padding)),
            "conv": nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(same_padding))
        })

        self.bn_choices = nn.ModuleDict({
            "bn": nn.BatchNorm2d(num_features=bn_channels),
            "bcn": norm.BCNorm(num_features=bn_channels, num_groups=32),
            "nope": nn.Identity()
        })

        self.act_choices = nn.ModuleDict({
            "relu": nn.ReLU(inplace=True),
            "mish": act.Mish(),
            "swish": act.Swish()
        })