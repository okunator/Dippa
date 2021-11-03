import torch
from typing import Union, List

from .._base._base_bottleneck import BaseBottleneckConv


class BottleneckDense(BaseBottleneckConv):
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
        Bottleneck dense conv block that can be used to build deep
        dense bottleneck layers.

        (DenseNet): Densely Connected Convolutional Networks
            - https://arxiv.org/abs/1608.06993

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
        super(BottleneckDense, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            pre_attend=True,
            preactivate=False,
            **kwargs
        )

    def forward(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]]
        ) -> torch.Tensor:
 
        # dense skip
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        # pre-attention
        out = self.attend(x)

        # bottleneck
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act(out)

        return out


class BottleneckDensePreact(BaseBottleneckConv):
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
        Preactivated bottleneck dense conv block that can be used to
        build deep preactivated dense bottleneck layers.

        (DenseNet): Densely Connected Convolutional Networks
            - https://arxiv.org/abs/1608.06993

        Preactivation introduced:
            - Identity Mappings in Deep Residual Networks:
                - https://arxiv.org/abs/1603.05027

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
        super(BottleneckDensePreact, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            pre_attend=True,
            preactivate=True,
            **kwargs
        )

    def forward(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]]
        ) -> torch.Tensor:
 
        # dense skip
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        # pre-attention
        out = self.attend(x)

        # bottleneck
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.conv3(out)
        out = self.act(out)

        return out