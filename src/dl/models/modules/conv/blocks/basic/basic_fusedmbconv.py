import torch

from .._base._base_fused_mbconv import BaseFusedMBConv


class FusedInvertedBasic(BaseFusedMBConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float=4.0,
            kernel_size: int=3,
            stride: int=1,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Fused mobile inverted basic conv block that can be used to 
        build deep residual basic FMBConv layers.

        Efficientnet-edgetpu: Creating accelerator-optimized neural networks with automl.
            - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

        EfficientNetV2: Smaller Models and Faster Training
            - https://arxiv.org/abs/2104.00298

        NOTE: 
        Basic in the name here means that no dense or residual skip 
        connections are applied in the forward method

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            expand_ratio (float, default=4.0):
                The ratio of channel expansion in the bottleneck
            kernel_size (int, default=3):
                The size of the convolution kernel.
            stride (int, default=1):
                Stride of the convolution operation
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
        super(FusedInvertedBasic, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            preactivate=False,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pointwise channel pooling conv
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        # attention
        out = self.attend(out)

        # Pointwise linear projection
        out = self.proj_conv(out)
        out = self.norm2(out)

        return out


class FusedInvertedBasicPreact(BaseFusedMBConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            expand_ratio: float=4.0,
            same_padding: bool=True,
            normalization: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            attention: str=None,
            **kwargs
        ) -> None:
        """
        Preactivated fused mobile inverted basic conv block that can 
        be used to build deep FMBconv basic layers with preactivation.


        (Preact-ResNet): Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027

        Efficientnet-edgetpu: Creating accelerator-optimized neural networks with automl.
            - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

        EfficientNetV2: Smaller Models and Faster Training
            - https://arxiv.org/abs/2104.00298

        NOTE: 
        Basic in the name here means that no dense or residual skip 
        connections are applied in the forward method

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            kernel_size (int, default=3):
                The size of the convolution kernel.
            expand_ratio (float, default=1.0):
                The ratio of channel expansion in the bottleneck
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
        super(FusedInvertedBasicPreact, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio,
            same_padding=same_padding,
            normalization=normalization,
            activation=activation,
            weight_standardize=weight_standardize,
            attention=attention,
            preactivate=True,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pointwise channel pooling conv
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        # attention
        out = self.attend(out)

        # Pointwise linear projection
        out = self.norm2(out)
        out = self.proj_conv(out)

        return out
        