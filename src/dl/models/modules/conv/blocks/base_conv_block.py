import torch.nn as nn

from ..ops import WSConv2d
from ...activations.utils import act_func
from ...normalization.utils import norm_func


class BaseConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            preactivate: bool=False
        ) -> None:
        """
        Base conv block that is used in all decoder blocks.
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
        """
        super(BaseConvBlock, self).__init__()
        
        self.normalization = normalization
        self.activation = activation
        self.conv_choice = "wsconv" if weight_standardize else "conv"

        # set norm channel number for preactivation or normal 
        norm_channels = in_channels if preactivate else out_channels
        
        # set convolution module
        if self.conv_choice == "wsconv":
            self.conv = WSConv2d(
                in_channels, out_channels, 
                kernel_size=3, padding=int(same_padding)
            )
        elif self.conv_choice == "conv":
            self.conv = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=3, padding=int(same_padding)
            )

        # set normalization module
        self.norm = norm_func(self.normalization, num_features=norm_channels)
            
        # set activation module
        self.act = act_func(self.activation)
