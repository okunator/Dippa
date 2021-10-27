import torch
import torch.nn as nn

from .base_conv_block import BaseConvBlock


class MBConvBlock(BaseConvBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            use_residual: bool=True,
            se_block: bool=True
        ) -> None:
        """
        Mobile inverted bottleneck Conv block.
        Squeeze and excitation-block can be added (optional)

        Common architectures using MBConv:

            - MobileNetV2: https://arxiv.org/abs/1801.04381
            - MnasNet: https://arxiv.org/abs/1807.11626
            - EfficientNet: https://arxiv.org/abs/1905.11946

        Squeeze and excitation:
            - SE: https://arxiv.org/abs/1709.01507

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
            use_residual (bool, default=True):
                If True, the identity is summed to the linear unit 
                before the final activation. (This param is used by
                the MultiMBBlock)
            se_block (bool, default=True):
                If True, a squeeze an SE-block is added
        """
        raise NotImplementedError