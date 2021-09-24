import torch
import torch.nn as nn

from ..base_conv_block import BaseConvBlock


class ResidualConvBlockPreact(BaseConvBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            use_residual: bool=True
        ) -> None:
        """
        Preactivated Residual conv block that can be used in decoders.
        Residual connection applied after the final conv.

        The forward method follows the preactivation implementation:
        https://arxiv.org/abs/1603.05027
        
         ------------------------------
        |                              |
        Input -> BN -> RELU -> CONV -> + -> OUTPUT

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Normalization method. One of: "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of (relu, swish. mish)
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            use_residual (bool, default=True):
                If True, the identity is summed to the linear unit 
                before the final activation
        """
        super(ResidualConvBlockPreact, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=True
        )
        self.use_residual = use_residual

        # Use channel pooling if dims don't match
        if in_channels != out_channels:
            self.add_module(
                "ch_pool", nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, padding=0, bias=False
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)

        if self.use_residual:
            if identity.shape[1] != x.shape[1]:
                identity = self.ch_pool(identity)
            x += identity

        return x


class ResidualConvBlock(BaseConvBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            use_residual: bool=True
        ) -> None:
        """
        Residual conv block that can be used in decoders.
        Residual connection applied before the final activation.

        The forward method follows the original implementation:
        https://arxiv.org/abs/1512.03385
        
         ----------------------
        |                      |
        Input -> CONV -> BN -> + -> RELU -> OUTPUT

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                if True, performs same-covolution
            batch_norm (str, default="bn"): 
                Normalization method. One of: "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of: "relu", "swish". "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            use_residual (bool, default=True):
                If True, the identity is summed to the linear unit 
                before the final activation
        """
        super(ResidualConvBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            same_padding=same_padding,
            batch_norm=batch_norm,
            activation=activation,
            weight_standardize=weight_standardize,
            preactivate=False
        )
        self.use_residual = use_residual

        # Use channel pooling if dims don't match
        if in_channels != out_channels:
            self.add_module(
                "ch_pool", nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, padding=0, bias=False
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = self.bn(x)

        if self.use_residual:
            if identity.shape[1] != x.shape[1]:
                identity = self.ch_pool(identity)
            x += identity

        x = self.act(x)
        return x


class MultiBlockResidual(nn.ModuleDict):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            same_padding: bool=True,
            batch_norm: str="bn",
            activation: str="relu",
            weight_standardize: bool=False,
            n_blocks: int=2,
            preactivate: bool=False
        ) -> None:
        """
        Stack residual conv blocks in a ModuleDict. These are used in 
        the full sized decoderblocks. The number of basic conv blocks 
        can be adjusted. Default is 2. The residual connection is 
        applied at the final conv block, before the last activation.

        Args:
        ----------
            in_channels (int):
                Number of input channels
            out_channels (int):
                Number of output channels
            same_padding (bool, default=True):
                If True, performs same-covolution
            batch_norm (str, default="bn"): 
                Normalization method. One of "bn", "bcn", None
            activation (str, default="relu"):
                Activation method. One of: "relu", "swish". "mish"
            weight_standardize (bool, default=False):
                If True, perform weight standardization
            n_blocks (int, default=2):
                Number of BasicConvBlocks used in this block
            preactivate (bool, default=False)
                If True, normalization and activation are applied before
                convolution
        """
        super(MultiBlockResidual, self).__init__()

        # use either preact or normal (original) resblock
        ResBlock = ResidualConvBlock
        if preactivate:
            ResBlock = ResidualConvBlockPreact

        # First res conv block. If n_blocks != 1 no residual skip 
        # at the first conv block
        use_residual = n_blocks == 1
        self.conv1 = ResBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            same_padding=same_padding,
            batch_norm=batch_norm, 
            activation=activation, 
            weight_standardize=weight_standardize,
            use_residual=use_residual
        )

        blocks = list(range(1, n_blocks))
        for i in blocks:
            # apply residual connection at the final conv block
            use_residual = i == blocks[-1]
            conv_block = ResBlock(
                in_channels=out_channels, 
                out_channels=out_channels, 
                same_padding=same_padding, 
                batch_norm=batch_norm, 
                activation=activation, 
                weight_standardize=weight_standardize,
                use_residual=use_residual
            )
            self.add_module('conv%d' % (i + 1), conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, conv_block in self.items():
            x = conv_block(x)
        return x